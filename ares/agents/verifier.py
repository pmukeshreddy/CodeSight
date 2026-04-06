from __future__ import annotations

import ast
import difflib
import hashlib
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ares.agents._llm import LLMAdapter
from ares.utils.json_utils import parse_llm_json

try:  # pragma: no cover - optional dependency
    import resource
except ImportError:  # pragma: no cover - optional dependency
    resource = None

try:  # pragma: no cover - optional dependency
    from tree_sitter import Language, Parser
except ImportError:  # pragma: no cover - optional dependency
    Language = None
    Parser = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_javascript as tsjavascript
except ImportError:  # pragma: no cover - optional dependency
    tsjavascript = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_typescript as tstypescript
except ImportError:  # pragma: no cover - optional dependency
    tstypescript = None


_FIX_BATCH_SIZE = 10
_REPRO_BATCH_SIZE = 10

BATCH_REPRO_SYSTEM_PROMPT = """
Generate minimal regression tests that reproduce the concrete bugs described in review comments.
You will receive multiple targets. For each target, return strict JSON in sections separated by markers.

Format:
--- target_0 ---
{"path": "test_file_path.py", "code": "import ...\\ndef test_...(): ..."}
--- target_1 ---
{"path": "test_file_path.py", "code": "import ...\\ndef test_...(): ..."}

Each test should fail on the original code and pass after the fix.
No prose outside the markers. Only JSON per target.
""".strip()

BATCH_FIX_SYSTEM_PROMPT = """
You generate the minimal code change required to fix concrete bugs.
You will receive multiple targets. For each target, return only the full fixed function source.
Separate each fix with the exact marker for that target.

Format:
--- target_0 ---
<fixed function source for target 0>
--- target_1 ---
<fixed function source for target 1>

No prose, no markdown fences. Only the fixed function source for each target.
""".strip()

VERIFY_CPU_LIMIT_SECONDS = int(os.getenv("ARES_VERIFY_CPU_SECONDS", "20"))
VERIFY_MEMORY_LIMIT_MB = int(os.getenv("ARES_VERIFY_MEMORY_MB", "1024"))
VERIFY_PROCESS_LIMIT = int(os.getenv("ARES_VERIFY_PROCESS_LIMIT", "32"))

_PYTEST_FLAGS = [
    "-q", "--tb=short", "--no-header",
    "--assert=plain",
    "-p", "no:cacheprovider",
    "-p", "no:warnings",
    "-p", "no:cov",
]


def _pytest_task(args: tuple) -> dict:
    """Run pytest in a forked worker. Captures stdout/stderr via a StringIO redirect."""
    import io
    import os
    import sys
    import pytest

    test_files, temp_dir, timeout = args
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    old_cwd = os.getcwd()
    try:
        sys.stdout = buf_out
        sys.stderr = buf_err
        os.chdir(temp_dir)
        rc = pytest.main([*_PYTEST_FLAGS, "--rootdir", temp_dir, *test_files])
    except Exception as exc:  # pragma: no cover
        buf_err.write(str(exc))
        rc = 1
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        os.chdir(old_cwd)
    stdout = buf_out.getvalue()
    stderr = buf_err.getvalue()
    success = rc == 0
    return {
        "command": f"pytest {' '.join(test_files)}",
        "returncode": int(rc),
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "error": "" if success else (stderr or stdout),
        "error_excerpt": (stderr or stdout)[:500] if not success else "",
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Process-level shared pytest pool — created once, reused by all Verifier
# instances (avoids spawning 16 idle workers when 4 PRs run in parallel).
# ---------------------------------------------------------------------------
_shared_pool: "multiprocessing.pool.Pool | None" = None
_shared_pool_lock = threading.Lock()


def _get_shared_pytest_pool() -> "multiprocessing.pool.Pool | None":
    global _shared_pool
    if _shared_pool is not None:
        return _shared_pool
    with _shared_pool_lock:
        if _shared_pool is not None:
            return _shared_pool
        try:
            ctx = multiprocessing.get_context("fork")
            _shared_pool = ctx.Pool(processes=4, maxtasksperchild=1)
            print("[verifier] shared pytest pool ready (fork, 4 workers)", flush=True)
        except Exception as exc:
            print(f"[verifier] pytest pool unavailable, using subprocess fallback: {exc}", flush=True)
            _shared_pool = None
    return _shared_pool


class Verifier:
    def __init__(
        self,
        api_key: str,
        repo_path: str,
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        cpu_limit_seconds: int = VERIFY_CPU_LIMIT_SECONDS,
        memory_limit_mb: int = VERIFY_MEMORY_LIMIT_MB,
        process_limit: int = VERIFY_PROCESS_LIMIT,
    ):
        self.client = LLMAdapter(api_key=api_key, model=model, provider=provider)
        self.model = model
        self.repo_path = os.path.abspath(repo_path)
        self.cpu_limit_seconds = max(1, int(cpu_limit_seconds))
        self.memory_limit_bytes = max(256, int(memory_limit_mb)) * 1024 * 1024
        self.process_limit = max(1, int(process_limit))
        self._parsers = self._build_parsers()
        self._fix_executor = ThreadPoolExecutor(max_workers=1)

    def _get_pytest_pool(self) -> "multiprocessing.pool.Pool | None":
        return _get_shared_pytest_pool()

    def verify_comments(self, candidates: list[dict], original_source: dict) -> list[dict]:
        return [
            result
            for result in self.verify_candidates(candidates, original_source)
            if result.get("verification", {}).get("status") == "passed"
        ]

    def prepare_pending(
        self,
        candidates: list[dict],
        verification_modes: list[str] | None = None,
        cache: dict[str, dict] | None = None,
    ) -> tuple[list[dict | None], list[tuple[int, dict, str, str]]]:
        """Separate cached/skipped results from items that need LLM + verification.

        Returns ``(partial_results, pending)`` where *partial_results* has the
        same length as *candidates* (``None`` for pending slots) and *pending*
        is the list of items that still require fix-generation and verification.
        """
        verification_cache = cache if cache is not None else {}
        results: list[dict | None] = [None] * len(candidates)
        pending: list[tuple[int, dict, str, str]] = []
        for index, candidate in enumerate(candidates):
            candidate_key = self._candidate_cache_key(candidate)
            requested_mode = self._requested_mode(verification_modes, index)
            cached = verification_cache.get(candidate_key)
            if cached is not None and self._can_reuse_cached(cached, requested_mode):
                results[index] = dict(cached)
                continue
            if requested_mode == "skip":
                result = self._skipped_result(candidate)
                cached_mode = self._mode_rank(cached.get("verification", {}).get("mode", "skip")) if cached else -1
                if self._mode_rank(result.get("verification", {}).get("mode", "skip")) >= cached_mode:
                    verification_cache[candidate_key] = dict(result)
                results[index] = result
                continue
            pending.append((index, candidate, candidate_key, requested_mode))
        return results, pending

    def build_fix_batch_requests(self, pending: list[tuple]) -> list[tuple[list[tuple], str, str, int]]:
        """Build fix-generation prompts for the batch API.

        Returns ``[(batch_items, system_prompt, user_prompt, max_tokens), ...]``.
        """
        if not pending:
            return []
        requests: list[tuple[list[tuple], str, str, int]] = []
        for i in range(0, len(pending), _FIX_BATCH_SIZE):
            batch = pending[i : i + _FIX_BATCH_SIZE]
            prompt_parts: list[str] = []
            for local_idx, (_, candidate, _key, _mode) in enumerate(batch):
                language = self._markdown_language(candidate["file"])
                prompt_parts.append(
                    f"--- target_{local_idx} ---\n"
                    f"Review comment:\n{candidate['comment']}\n\n"
                    f"Code to fix:\n```{language}\n{candidate.get('function_source', '')}\n```"
                )
            combined = "\n\n".join(prompt_parts)
            max_tokens = min(1000 * len(batch), 16384)
            requests.append((batch, BATCH_FIX_SYSTEM_PROMPT, combined, max_tokens))
        return requests

    def parse_fix_batch_responses(
        self,
        requests: list[tuple[list[tuple], str, str, int]],
        responses: list[str],
    ) -> dict[int, str]:
        """Parse fix-generation batch responses back into ``{candidate_index: fixed_source}``."""
        results: dict[int, str] = {}
        for (batch, _, _, _), raw in zip(requests, responses):
            parsed = self._parse_fix_batch_response(raw, len(batch))
            for local_idx, (index, _cand, _key, _mode) in enumerate(batch):
                results[index] = parsed.get(local_idx, "")
        return results

    def build_repro_batch_requests(
        self, full_pending: list[tuple],
    ) -> list[tuple[list[tuple], list[tuple[int, str, str, str]], str, str, int]]:
        """Build repro-test-generation prompts for the batch API.

        Returns ``[(batch_items, meta, system_prompt, user_prompt, max_tokens), ...]``.
        """
        if not full_pending:
            return []
        requests: list[tuple[list[tuple], list[tuple[int, str, str, str]], str, str, int]] = []
        for i in range(0, len(full_pending), _REPRO_BATCH_SIZE):
            batch = full_pending[i : i + _REPRO_BATCH_SIZE]
            prompt_parts: list[str] = []
            meta: list[tuple[int, str, str, str]] = []
            for local_idx, (idx, candidate, _key, _mode) in enumerate(batch):
                filepath = candidate.get("file", "")
                ext = Path(filepath).suffix.lower()
                if ext == ".py":
                    default_path = f".ares/generated_tests/test_{Path(filepath).stem}_{self._candidate_cache_key(candidate)[:8]}.py"
                    framework = "pytest"
                    import_hint = Path(filepath).with_suffix("").as_posix().replace("/", ".")
                elif ext in {".js", ".jsx", ".ts", ".tsx"}:
                    stem = Path(filepath).stem
                    default_path = f".ares/generated_tests/{stem}_{self._candidate_cache_key(candidate)[:8]}.test.js"
                    framework = "jest"
                    import_hint = os.path.relpath(filepath, ".ares/generated_tests").replace(os.sep, "/")
                    if not import_hint.startswith("."):
                        import_hint = f"./{import_hint}"
                    if import_hint.endswith((".ts", ".tsx", ".js", ".jsx")):
                        import_hint = import_hint.rsplit(".", 1)[0]
                else:
                    meta.append((idx, "", "", ""))
                    prompt_parts.append(f"--- target_{local_idx} ---\nSkip — unsupported language.")
                    continue
                meta.append((idx, default_path, framework, import_hint))
                prompt_parts.append(
                    f"--- target_{local_idx} ---\n"
                    f"Language: {self._markdown_language(filepath) or ext}\n"
                    f"Framework: {framework}\n"
                    f"Default test path: {default_path}\n"
                    f"Import hint: {import_hint}\n"
                    f"Review comment: {candidate.get('comment', '')}\n"
                    f"Function signature: {candidate.get('function_signature', '')}\n"
                    f"Function source:\n```{self._markdown_language(filepath)}\n{candidate.get('function_source', '')}\n```"
                )
            combined = "\n\n".join(prompt_parts)
            max_tokens = min(500 * len(batch), 10240)
            requests.append((batch, meta, BATCH_REPRO_SYSTEM_PROMPT, combined, max_tokens))
        return requests

    def parse_repro_batch_responses(
        self,
        requests: list[tuple[list[tuple], list[tuple[int, str, str, str]], str, str, int]],
        responses: list[str],
    ) -> dict[int, dict | None]:
        """Parse repro-test batch responses back into ``{candidate_index: repro_dict_or_None}``."""
        results: dict[int, dict | None] = {}
        for (batch, meta, _, _, _), raw in zip(requests, responses):
            parsed_sections = self._parse_fix_batch_response(raw, len(batch))
            for local_idx, (idx, default_path, framework, import_hint) in enumerate(meta):
                if not default_path:
                    results[idx] = None
                    continue
                section = parsed_sections.get(local_idx, "")
                if not section.strip():
                    results[idx] = None
                    continue
                payload = parse_llm_json(section)
                code = str(payload.get("code", "")).strip()
                path = str(payload.get("path", default_path)).strip() or default_path
                if not code:
                    results[idx] = None
                    continue
                results[idx] = {"path": path, "code": code, "framework": framework, "import_hint": import_hint}
        return results

    def verify_candidates(
        self,
        candidates: list[dict],
        original_source: dict,
        cache: dict[str, dict] | None = None,
        verification_modes: list[str] | None = None,
        pre_fixes: dict[int, str] | None = None,
        pre_repros: dict[int, dict | None] | None = None,
    ) -> list[dict]:
        verification_cache = cache if cache is not None else {}
        results, pending = self.prepare_pending(candidates, verification_modes, verification_cache)
        if not pending:
            return [result for result in results if result is not None]
        # Use provided pre-computed LLM results or generate them now.
        if pre_fixes is None:
            pre_fixes = self._batch_generate_fixes(pending)
        if pre_repros is None:
            full_pending = [item for item in pending if item[3] == "full"]
            pre_repros = self._batch_generate_repro_tests(full_pending) if full_pending else {}
        base_temp = self._create_temp_repo()
        try:
            max_workers = min(len(pending), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._verify_one_pending, item, original_source, base_temp,
                        pre_fix=pre_fixes.get(item[0], ""),
                        pre_repro=pre_repros.get(item[0]),
                    ): item
                    for item in pending
                }
                for future in as_completed(futures):
                    index, candidate_key, result = future.result()
                    verification_cache[candidate_key] = dict(result)
                    results[index] = result
        finally:
            shutil.rmtree(os.path.dirname(base_temp), ignore_errors=True)
        return [result for result in results if result is not None]

    def _verify_one_pending(
        self,
        item: tuple,
        original_source: dict,
        base_temp: str,
        pre_fix: str = "",
        pre_repro: dict | None = None,
    ) -> tuple:
        index, candidate, candidate_key, mode = item
        candidate_outer = tempfile.mkdtemp(prefix="ares-verify-cand-")
        try:
            temp_root = self._hardlink_tree(base_temp, os.path.join(candidate_outer, "repo"))
            result = self._verify_single(candidate, original_source, temp_root, mode, pre_fix=pre_fix, pre_repro=pre_repro)
        finally:
            shutil.rmtree(candidate_outer, ignore_errors=True)
        return index, candidate_key, result

    def verify_bundle(self, candidates: list[dict], original_source: dict) -> dict:
        working = [dict(candidate) for candidate in candidates]
        if len(working) <= 1:
            reason = "Bundle verification requires at least two eligible comments."
            survivors = [
                self._annotate_bundle_result(candidate, "skipped", reason=reason, group_size=len(working))
                for candidate in working
            ]
            return {"survivors": survivors, "dropped": [], "group_size": len(working)}
        dropped: list[dict] = []
        group_size = len(working)
        temp_root = self._create_temp_repo()
        try:
            while working:
                validation = self._run_bundle_validation(temp_root, working, original_source)
                if validation.get("status") == "passed":
                    survivors = [
                        self._annotate_bundle_result(
                            candidate,
                            "passed",
                            reason=validation.get("reason", "Bundle verification passed."),
                            group_size=group_size,
                            details=validation,
                        )
                        for candidate in working
                    ]
                    return {"survivors": survivors, "dropped": dropped, "group_size": group_size}
                if len(working) == 1:
                    dropped.append(
                        self._annotate_bundle_result(
                            working[0],
                            "failed",
                            reason=validation.get("reason", "Combined patch did not validate."),
                            group_size=group_size,
                            details=validation,
                        )
                    )
                    return {"survivors": [], "dropped": dropped, "group_size": group_size}
                drop_index = self._choose_bundle_drop_index(working, validation)
                removed = working.pop(drop_index)
                dropped.append(
                    self._annotate_bundle_result(
                        removed,
                        "failed",
                        reason=validation.get("reason", "Combined patch did not validate."),
                        group_size=group_size,
                        details=validation,
                    )
                )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
        return {"survivors": [], "dropped": dropped, "group_size": group_size}

    def _verify_single(self, candidate: dict, original_source: dict, temp_root: str, mode: str = "full", pre_fix: str = "", pre_repro: dict | None = None) -> dict:
        filepath = candidate["file"]
        file_source = original_source.get(filepath)
        if not file_source or not candidate.get("function_source"):
            return self._failed_result(
                candidate,
                reason="Missing file source or function source for verification.",
                status="inconclusive",
                mode=mode,
            )
        import time as _time
        _t0 = _time.monotonic()
        fix_attempt = self._attempt_fix_generation(candidate, filepath, candidate["function_source"], original_source, temp_root, mode, pre_fix=pre_fix)
        _t_fix = _time.monotonic()
        fixed_source = fix_attempt["fixed_source"]
        if not fixed_source.strip() or fixed_source.strip() == candidate["function_source"].strip():
            return self._failed_result(
                candidate,
                reason="Generated fix was empty or identical to the existing function.",
                status="inconclusive",
                mode=mode,
            )
        ext = Path(filepath).suffix
        ast_change_type = self._ast_change_type(candidate["function_source"], fixed_source, ext)
        new_file_source = self._replace_lines(
            file_source,
            candidate["function_line_start"],
            candidate["function_line_end"],
            fixed_source,
        )
        validation = (
            fix_attempt["validation"]
            if mode == "compile" and fix_attempt["validation"] is not None
            else self._run_validation(
            temp_root,
            candidate,
            filepath,
            new_file_source,
            file_source,
            candidate.get("test_files", []),
            mode=mode,
            pre_repro=pre_repro,
            )
        )
        _t_val = _time.monotonic()
        print(f"[timing]           verify_single fix_gen={_t_fix-_t0:.2f}s validation={_t_val-_t_fix:.2f}s mode={mode}", flush=True)
        status = self._verification_status(validation, ast_change_type, mode)
        verification_key = self._verification_key(candidate["comment"], fixed_source)
        reason = self._verification_reason(validation, ast_change_type, status, mode)
        return {
            **candidate,
            "suggested_code": fixed_source,
            "fix_diff": self._generate_unified_diff(file_source, new_file_source, filepath),
            "validation": validation,
            "generation": fix_attempt["generation"],
            "verification_key": verification_key,
            "verification_reason": reason,
            "verification": {
                "status": status,
                "mode": mode,
                "ast_change_type": ast_change_type,
                "reason": reason,
                "candidate_key": self._candidate_cache_key(candidate),
            },
        }

    def _generate_fix(self, filepath: str, file_source: str, comment: str, prior_error: str = "") -> str:
        if not self.client.available:
            return ""
        system_prompt = (
            "You generate the minimal code change required to fix a concrete bug. "
            "Return only the full fixed function source, with no prose and no markdown."
        )
        language = self._markdown_language(filepath)
        retry_hint = ""
        if prior_error.strip():
            retry_hint = f"\n\nPrevious attempt failed. Correct the fix and avoid this error:\n{prior_error.strip()}"
        user_prompt = (
            f"Review comment:\n{comment}\n\nCode to fix:\n```{language}\n{file_source}\n```{retry_hint}"
        )
        future = self._fix_executor.submit(self.client.complete, system_prompt, user_prompt, 1000)
        try:
            return future.result(timeout=45).strip()
        except TimeoutError:
            return ""

    def _batch_generate_fixes(self, pending: list[tuple]) -> dict[int, str]:
        """Pre-generate fixes for pending candidates in batched LLM calls.

        Returns ``{candidate_index: fixed_source}``."""
        if not self.client.available or not pending:
            return {}
        results: dict[int, str] = {}
        batches: list[list[tuple]] = []
        for i in range(0, len(pending), _FIX_BATCH_SIZE):
            batches.append(pending[i : i + _FIX_BATCH_SIZE])
        for batch in batches:
            prompt_parts: list[str] = []
            for local_idx, (_, candidate, _key, _mode) in enumerate(batch):
                language = self._markdown_language(candidate["file"])
                prompt_parts.append(
                    f"--- target_{local_idx} ---\n"
                    f"Review comment:\n{candidate['comment']}\n\n"
                    f"Code to fix:\n```{language}\n{candidate.get('function_source', '')}\n```"
                )
            combined = "\n\n".join(prompt_parts)
            max_tokens = min(1000 * len(batch), 16384)
            raw = self.client.complete(BATCH_FIX_SYSTEM_PROMPT, combined, max_tokens)
            parsed = self._parse_fix_batch_response(raw, len(batch))
            for local_idx, (index, _cand, _key, _mode) in enumerate(batch):
                results[index] = parsed.get(local_idx, "")
        return results

    def _parse_fix_batch_response(self, raw: str, count: int) -> dict[int, str]:
        """Parse batched fix response by splitting on ``--- target_N ---`` markers."""
        results: dict[int, str] = {}
        parts = re.split(r"---\s*target_(\d+)\s*---", raw)
        # parts: [preamble, "0", code_0, "1", code_1, ...]
        for i in range(1, len(parts) - 1, 2):
            try:
                target_idx = int(parts[i])
            except ValueError:
                continue
            code = parts[i + 1].strip()
            # Strip markdown fences the LLM may add despite instructions.
            lines = code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            results[target_idx] = "\n".join(lines)
        return results

    def _batch_generate_repro_tests(self, full_pending: list[tuple]) -> dict[int, dict | None]:
        """Pre-generate repro tests for full-mode candidates in batched LLM calls.

        Returns ``{candidate_index: repro_dict_or_None}``."""
        if not self.client.available or not full_pending:
            return {}
        results: dict[int, dict | None] = {}
        batches: list[list[tuple]] = []
        for i in range(0, len(full_pending), _REPRO_BATCH_SIZE):
            batches.append(full_pending[i : i + _REPRO_BATCH_SIZE])
        for batch in batches:
            prompt_parts: list[str] = []
            meta: list[tuple[int, str, str, str]] = []  # (index, default_path, framework, import_hint)
            for local_idx, (idx, candidate, _key, _mode) in enumerate(batch):
                filepath = candidate.get("file", "")
                ext = Path(filepath).suffix.lower()
                if ext == ".py":
                    default_path = f".ares/generated_tests/test_{Path(filepath).stem}_{self._candidate_cache_key(candidate)[:8]}.py"
                    framework = "pytest"
                    import_hint = Path(filepath).with_suffix("").as_posix().replace("/", ".")
                elif ext in {".js", ".jsx", ".ts", ".tsx"}:
                    stem = Path(filepath).stem
                    default_path = f".ares/generated_tests/{stem}_{self._candidate_cache_key(candidate)[:8]}.test.js"
                    framework = "jest"
                    import_hint = os.path.relpath(filepath, ".ares/generated_tests").replace(os.sep, "/")
                    if not import_hint.startswith("."):
                        import_hint = f"./{import_hint}"
                    if import_hint.endswith((".ts", ".tsx", ".js", ".jsx")):
                        import_hint = import_hint.rsplit(".", 1)[0]
                else:
                    meta.append((idx, "", "", ""))
                    prompt_parts.append(f"--- target_{local_idx} ---\nSkip — unsupported language.")
                    continue
                meta.append((idx, default_path, framework, import_hint))
                prompt_parts.append(
                    f"--- target_{local_idx} ---\n"
                    f"Language: {self._markdown_language(filepath) or ext}\n"
                    f"Framework: {framework}\n"
                    f"Default test path: {default_path}\n"
                    f"Import hint: {import_hint}\n"
                    f"Review comment: {candidate.get('comment', '')}\n"
                    f"Function signature: {candidate.get('function_signature', '')}\n"
                    f"Function source:\n```{self._markdown_language(filepath)}\n{candidate.get('function_source', '')}\n```"
                )
            combined = "\n\n".join(prompt_parts)
            max_tokens = min(500 * len(batch), 10240)
            raw = self.client.complete(BATCH_REPRO_SYSTEM_PROMPT, combined, max_tokens)
            parsed_sections = self._parse_fix_batch_response(raw, len(batch))
            for local_idx, (idx, default_path, framework, import_hint) in enumerate(meta):
                if not default_path:
                    results[idx] = None
                    continue
                section = parsed_sections.get(local_idx, "")
                if not section.strip():
                    results[idx] = None
                    continue
                payload = parse_llm_json(section)
                code = str(payload.get("code", "")).strip()
                path = str(payload.get("path", default_path)).strip() or default_path
                if not code:
                    results[idx] = None
                    continue
                results[idx] = {"path": path, "code": code, "framework": framework, "import_hint": import_hint}
        return results

    def _attempt_fix_generation(
        self,
        candidate: dict,
        filepath: str,
        function_source: str,
        original_source: dict,
        temp_root: str,
        mode: str,
        pre_fix: str = "",
    ) -> dict:
        generation = {"attempts": 0, "retried_for": "", "last_error_type": "", "last_error": ""}
        if mode == "skip":
            return {"fixed_source": "", "validation": None, "generation": generation}
        fixed_source = pre_fix.strip() if pre_fix.strip() else self._generate_fix(filepath, function_source, candidate["comment"])
        generation["attempts"] = 1
        if not fixed_source.strip() or fixed_source.strip() == function_source.strip():
            return {"fixed_source": fixed_source, "validation": None, "generation": generation}
        validation = self._preview_validation(candidate, original_source, temp_root, fixed_source)
        compile_error_type = validation.get("compile", {}).get("error_type", "")
        generation["last_error_type"] = compile_error_type
        generation["last_error"] = validation.get("compile_error", "")
        if compile_error_type not in {"SyntaxError", "IndentationError", "ParseError"}:
            return {"fixed_source": fixed_source, "validation": validation, "generation": generation}
        retry_error = validation.get("compile_error", "") or "Syntax error in generated fix."
        retried_source = self._generate_fix(filepath, function_source, candidate["comment"], prior_error=retry_error)
        generation["attempts"] = 2
        generation["retried_for"] = compile_error_type
        if not retried_source.strip() or retried_source.strip() == function_source.strip():
            return {"fixed_source": fixed_source, "validation": validation, "generation": generation}
        retried_validation = self._preview_validation(candidate, original_source, temp_root, retried_source)
        generation["last_error_type"] = retried_validation.get("compile", {}).get("error_type", "")
        generation["last_error"] = retried_validation.get("compile_error", "")
        if retried_validation.get("compiles", False):
            return {"fixed_source": retried_source, "validation": retried_validation, "generation": generation}
        return {"fixed_source": fixed_source, "validation": validation, "generation": generation}

    def _preview_validation(
        self,
        candidate: dict,
        original_source: dict,
        temp_root: str,
        fixed_source: str,
    ) -> dict:
        filepath = candidate["file"]
        file_source = original_source.get(filepath, "")
        new_file_source = self._replace_lines(
            file_source,
            candidate["function_line_start"],
            candidate["function_line_end"],
            fixed_source,
        )
        return self._run_validation(
            temp_root,
            candidate,
            filepath,
            new_file_source,
            file_source,
            candidate.get("test_files", []),
            mode="compile",
        )

    def _markdown_language(self, filepath: str) -> str:
        extension = Path(filepath).suffix.lower()
        if extension == ".py":
            return "python"
        if extension in {".ts", ".tsx"}:
            return "typescript"
        if extension in {".js", ".jsx"}:
            return "javascript"
        return ""

    @staticmethod
    def _link_or_copy(s: str, d: str) -> None:
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    def _create_temp_repo(self) -> str:
        temp_root = tempfile.mkdtemp(prefix="ares-verify-")
        repo_copy = os.path.join(temp_root, "repo")
        shutil.copytree(
            self.repo_path,
            repo_copy,
            ignore=shutil.ignore_patterns(".git", "node_modules", ".venv", "__pycache__", "dist", "build"),
            copy_function=self._link_or_copy,
        )
        return repo_copy

    def _hardlink_tree(self, src: str, dst: str) -> str:
        """Copy *src* tree to *dst* using hardlinks — O(metadata) not O(data)."""
        shutil.copytree(src, dst, copy_function=self._link_or_copy)
        return dst

    def _safe_write(self, path: "Path", content: str) -> None:
        """Write *content* to *path*, unlinking first to break any hardlink."""
        if path.exists():
            path.unlink()
        path.write_text(content, encoding="utf-8")

    def _run_validation(
        self,
        temp_dir: str,
        candidate: dict,
        filepath: str,
        fixed_file_source: str,
        original_file_source: str,
        test_files: list[str],
        mode: str = "full",
        pre_repro: dict | None = None,
    ) -> dict:
        target = Path(temp_dir) / filepath
        existing_source = (
            target.read_text(encoding="utf-8", errors="ignore")
            if target.exists()
            else original_file_source
        )
        generated_test = pre_repro if pre_repro is not None else (self._generate_repro_test(candidate) if mode == "full" else "")
        generated_test_path = self._materialize_generated_test(temp_dir, generated_test) if generated_test else None
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            repro = self._empty_repro_result(generated_test_path, generated_test)
            if mode == "full" and generated_test_path:
                self._safe_write(target, original_file_source)
                repro_before = self._run_generated_test(temp_dir, generated_test_path, generated_test)
                repro["before"] = repro_before
                repro["attempted"] = True
                repro["bug_reproduced"] = self._repro_failed(repro_before)
            self._safe_write(target, fixed_file_source)
            validation = self._run_validation_commands(temp_dir, filepath, test_files, mode=mode)
            if mode == "full" and generated_test_path and validation.get("compiles", False):
                repro_after = self._run_generated_test(temp_dir, generated_test_path, generated_test)
                repro["after"] = repro_after
                repro["fixed"] = repro_after.get("success", False)
                repro["status"] = self._repro_status(repro)
            validation["repro"] = repro
            return validation
        finally:
            self._safe_write(target, existing_source)
            if generated_test_path and generated_test_path.exists():
                generated_test_path.unlink()

    def _run_validation_commands(self, temp_dir: str, filepath: str, test_files: list[str], mode: str = "full") -> dict:
        ext = Path(filepath).suffix
        compiles = True
        tests_pass = True
        executed_tests: list[str] = []
        compile_details = self._empty_command_result()
        tests_details = self._empty_test_result()
        if ext == ".py":
            compile_result = self._run_command(
                [sys.executable, "-m", "py_compile", filepath], cwd=temp_dir
            )
            compile_details = {
                **compile_result,
                "attempted": True,
                "error_type": self._classify_error(compile_result, phase="compile"),
            }
            compiles = compile_result["success"]
            if mode == "full" and compiles and test_files:
                executed_tests = [path for path in test_files if (Path(temp_dir) / path).exists()]
                if executed_tests:
                    test_result = self._run_pytest(executed_tests, cwd=temp_dir, timeout=60)
                    tests_details = {
                        **test_result,
                        "attempted": True,
                        "executed": executed_tests,
                        "error_type": self._classify_error(test_result, phase="test"),
                        **self._extract_test_summary(test_result, executed_tests),
                    }
                    tests_pass = test_result["success"]
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            package_json = Path(temp_dir) / "package.json"
            if package_json.exists():
                if ext in {".ts", ".tsx"}:
                    compile_result = self._run_command(["npx", "tsc", "--noEmit"], cwd=temp_dir, timeout=60)
                    compile_details = {
                        **compile_result,
                        "attempted": True,
                        "error_type": self._classify_error(compile_result, phase="compile"),
                    }
                    compiles = compile_result["success"]
                elif ext in {".js", ".jsx"}:
                    compile_details = {**self._empty_command_result(), "attempted": False, "success": True, "error_type": ""}
                if mode == "full" and compiles and test_files:
                    executed_tests = [path for path in test_files if (Path(temp_dir) / path).exists()]
                    if executed_tests:
                        test_result = self._run_command(
                            ["npx", "jest", *executed_tests, "--runInBand"],
                            cwd=temp_dir,
                            timeout=60,
                        )
                        tests_details = {
                            **test_result,
                            "attempted": True,
                            "executed": executed_tests,
                            "error_type": self._classify_error(test_result, phase="test"),
                            **self._extract_test_summary(test_result, executed_tests),
                        }
                        tests_pass = test_result["success"]
        compile_error = compile_details.get("error_excerpt", "")
        test_error = tests_details.get("error_excerpt", "")
        return {
            "compiles": compiles,
            "tests_pass": tests_pass,
            "compile_attempted": bool(compile_details.get("attempted")),
            "tests_attempted": bool(tests_details.get("attempted")),
            "test_results": executed_tests,
            "test_pass_ratio": tests_details.get("pass_ratio"),
            "compile_error": compile_error,
            "test_error": test_error,
            "compile_error_type": compile_details.get("error_type", ""),
            "test_error_type": tests_details.get("error_type", ""),
            "compile": compile_details,
            "tests": tests_details,
        }

    def _run_bundle_validation(self, temp_dir: str, candidates: list[dict], original_source: dict) -> dict:
        bundle_sources, conflicts = self._bundle_file_sources(candidates, original_source)
        if conflicts:
            conflict_keys = [
                candidate.get("verification_key") or candidate.get("verification", {}).get("candidate_key", "")
                for conflict in conflicts
                for candidate in conflict.get("candidates", [])
            ]
            conflict_paths = ", ".join(sorted({conflict.get("file", "") for conflict in conflicts if conflict.get("file")}))
            reason = f"Combined patch has overlapping edits in {conflict_paths or 'the same function'}."
            return {
                "status": "failed",
                "compiles": False,
                "tests_pass": False,
                "test_results": [],
                "compile_error": reason,
                "test_error": "",
                "compile_error_type": "BundleConflict",
                "test_error_type": "",
                "compile": {**self._empty_command_result(), "attempted": False, "error_type": "BundleConflict"},
                "tests": self._empty_test_result(),
                "conflicting_candidate_keys": [key for key in conflict_keys if key],
                "reason": reason,
            }
        backups = {}
        touched_files = []
        try:
            for filepath, updated_source in bundle_sources.items():
                target = Path(temp_dir) / filepath
                existing_source = (
                    target.read_text(encoding="utf-8", errors="ignore")
                    if target.exists()
                    else original_source.get(filepath, "")
                )
                backups[filepath] = existing_source
                target.parent.mkdir(parents=True, exist_ok=True)
                self._safe_write(target, updated_source)
                touched_files.append(filepath)
            validation = self._run_repo_validation_commands(
                temp_dir,
                touched_files=touched_files,
                test_files=self._bundle_test_files(candidates),
            )
            validation["status"] = "passed" if validation.get("compiles") and validation.get("tests_pass") else "failed"
            validation["conflicting_candidate_keys"] = []
            validation["reason"] = self._bundle_reason(validation)
            return validation
        finally:
            for filepath, previous_source in backups.items():
                target = Path(temp_dir) / filepath
                target.parent.mkdir(parents=True, exist_ok=True)
                self._safe_write(target, previous_source)

    def _run_repo_validation_commands(self, temp_dir: str, touched_files: list[str], test_files: list[str]) -> dict:
        compile_result = self._empty_command_result()
        tests_result = self._empty_test_result()
        compiles = True
        tests_pass = True
        python_files = [path for path in touched_files if Path(path).suffix == ".py"]
        typescript_files = [path for path in touched_files if Path(path).suffix in {".ts", ".tsx"}]
        package_json = Path(temp_dir) / "package.json"
        if python_files:
            command_result = self._run_command([sys.executable, "-m", "py_compile", *python_files], cwd=temp_dir)
            compile_result = {
                **command_result,
                "attempted": True,
                "error_type": self._classify_error(command_result, phase="compile"),
            }
            compiles = command_result["success"]
        elif typescript_files and package_json.exists():
            command_result = self._run_command(["npx", "tsc", "--noEmit"], cwd=temp_dir, timeout=60)
            compile_result = {
                **command_result,
                "attempted": True,
                "error_type": self._classify_error(command_result, phase="compile"),
            }
            compiles = command_result["success"]
        py_tests = [path for path in test_files if Path(path).suffix == ".py" and (Path(temp_dir) / path).exists()]
        js_tests = [path for path in test_files if Path(path).suffix in {".js", ".jsx", ".ts", ".tsx"} and (Path(temp_dir) / path).exists()]
        executed_tests = py_tests + js_tests
        test_command_results: list[dict] = []
        if compiles and py_tests:
            pytest_result = self._run_pytest(py_tests, cwd=temp_dir, timeout=60)
            test_command_results.append(
                {
                    **pytest_result,
                    "attempted": True,
                    "executed": py_tests,
                    "error_type": self._classify_error(pytest_result, phase="test"),
                    **self._extract_test_summary(pytest_result, py_tests),
                }
            )
            tests_pass = tests_pass and pytest_result["success"]
        if compiles and js_tests:
            jest_result = self._run_command(["npx", "jest", *js_tests, "--runInBand"], cwd=temp_dir, timeout=60)
            test_command_results.append(
                {
                    **jest_result,
                    "attempted": True,
                    "executed": js_tests,
                    "error_type": self._classify_error(jest_result, phase="test"),
                    **self._extract_test_summary(jest_result, js_tests),
                }
            )
            tests_pass = tests_pass and jest_result["success"]
        if test_command_results:
            tests_result = self._merge_test_results(test_command_results)
        return {
            "compiles": compiles,
            "tests_pass": tests_pass,
            "compile_attempted": bool(compile_result.get("attempted")),
            "tests_attempted": bool(tests_result.get("attempted")),
            "test_results": executed_tests,
            "test_pass_ratio": tests_result.get("pass_ratio"),
            "compile_error": compile_result.get("error_excerpt", ""),
            "test_error": tests_result.get("error_excerpt", ""),
            "compile_error_type": compile_result.get("error_type", ""),
            "test_error_type": tests_result.get("error_type", ""),
            "compile": compile_result,
            "tests": tests_result,
        }

    def _merge_test_results(self, results: list[dict]) -> dict:
        merged = self._empty_test_result()
        if not results:
            return merged
        merged["attempted"] = True
        merged["executed"] = [path for result in results for path in result.get("executed", [])]
        merged["passed_count"] = sum(int(result.get("passed_count", 0)) for result in results)
        merged["failed_count"] = sum(int(result.get("failed_count", 0)) for result in results)
        merged["error_count"] = sum(int(result.get("error_count", 0)) for result in results)
        merged["total_count"] = sum(int(result.get("total_count", 0)) for result in results)
        merged["success"] = all(bool(result.get("success")) for result in results)
        merged["command"] = " && ".join(result.get("command", "") for result in results if result.get("command"))
        merged["stdout"] = "\n".join(result.get("stdout", "") for result in results if result.get("stdout"))
        merged["stderr"] = "\n".join(result.get("stderr", "") for result in results if result.get("stderr"))
        merged["error"] = "\n".join(result.get("error", "") for result in results if result.get("error"))
        merged["error_excerpt"] = self._truncate_output(merged["error"] or merged["stderr"] or merged["stdout"])
        merged["returncode"] = 0 if merged["success"] else next(
            (result.get("returncode") for result in results if result.get("returncode") not in (0, None)),
            None,
        )
        merged["timed_out"] = any(bool(result.get("timed_out")) for result in results)
        merged["error_type"] = next((result.get("error_type", "") for result in results if result.get("error_type")), "")
        if merged["total_count"]:
            merged["pass_ratio"] = round(merged["passed_count"] / merged["total_count"], 3)
        return merged

    def _run_pytest(self, test_files: list[str], cwd: str, timeout: int = 60) -> dict:
        """Run pytest via the shared forked pool (fast) or fall back to subprocess."""
        pool = self._get_pytest_pool()
        if pool is not None:
            try:
                return pool.apply(
                    _pytest_task,
                    args=((test_files, cwd, timeout),),
                )
            except Exception:
                pass
        # fallback: subprocess
        return self._run_command(
            [sys.executable, "-m", "pytest", *_PYTEST_FLAGS, *test_files],
            cwd=cwd,
            timeout=timeout,
        )

    def _run_command(self, args: list[str], cwd: str, timeout: int = 15) -> dict:
        command = " ".join(args)
        process = None
        try:
            process = subprocess.Popen(
                args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=self._resource_limited_preexec if resource is not None else None,
            )
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    process.kill()
                stdout, stderr = process.communicate()
            error = f"Command timed out: {command}"
            return {
                "success": False,
                "command": command,
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "error": error,
                "error_excerpt": self._truncate_output(stderr or stdout or error),
                "timed_out": True,
            }
        except OSError:
            error = f"Command failed to run: {command}"
            return {
                "success": False,
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": error,
                "error_excerpt": error,
                "timed_out": False,
            }
        returncode = process.returncode if process is not None else None
        error = (stderr or stdout or "").strip()
        return {
            "success": returncode == 0,
            "command": command,
            "returncode": returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "error": error,
            "error_excerpt": self._truncate_output(error),
            "timed_out": False,
        }

    def _resource_limited_preexec(self) -> None:
        if resource is None:  # pragma: no cover - platform dependent
            return
        limits = [
            (getattr(resource, "RLIMIT_CPU", None), (self.cpu_limit_seconds, self.cpu_limit_seconds + 1)),
            (getattr(resource, "RLIMIT_AS", None), (self.memory_limit_bytes, self.memory_limit_bytes)),
            (getattr(resource, "RLIMIT_FSIZE", None), (10 * 1024 * 1024, 10 * 1024 * 1024)),
            (getattr(resource, "RLIMIT_NPROC", None), (self.process_limit, self.process_limit)),
        ]
        for limit_name, values in limits:
            if limit_name is None:
                continue
            try:
                resource.setrlimit(limit_name, values)
            except (OSError, ValueError):
                continue

    def _classify_error(self, result: dict, phase: str) -> str:
        if not result.get("attempted", True):
            return ""
        if result.get("timed_out"):
            return "TimeoutError"
        text = "\n".join(
            part for part in (result.get("stderr", ""), result.get("stdout", ""), result.get("error", "")) if part
        )
        patterns = [
            (r"\bIndentationError\b", "IndentationError"),
            (r"\bSyntaxError\b|TS\d+: error", "SyntaxError"),
            (r"\bTypeError\b", "TypeError"),
            (r"\bAssertionError\b|FAILED\b", "AssertionError"),
            (r"\bImportError\b|\bModuleNotFoundError\b|Cannot find module", "ImportError"),
            (r"\bMemoryError\b|out of memory|ENOMEM|killed", "ResourceError"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return label
        if not result.get("success", True):
            return "CompileError" if phase == "compile" else "ExecutionError"
        return ""

    def _generate_repro_test(self, candidate: dict) -> dict | None:
        if not self.client.available:
            return None
        filepath = candidate.get("file", "")
        ext = Path(filepath).suffix.lower()
        if ext == ".py":
            default_path = f".ares/generated_tests/test_{Path(filepath).stem}_{self._candidate_cache_key(candidate)[:8]}.py"
            framework = "pytest"
            import_hint = Path(filepath).with_suffix("").as_posix().replace("/", ".")
        elif ext in {".js", ".jsx", ".ts", ".tsx"}:
            stem = Path(filepath).stem
            default_path = f".ares/generated_tests/{stem}_{self._candidate_cache_key(candidate)[:8]}.test.js"
            framework = "jest"
            import_hint = os.path.relpath(filepath, ".ares/generated_tests").replace(os.sep, "/")
            if not import_hint.startswith("."):
                import_hint = f"./{import_hint}"
            if import_hint.endswith((".ts", ".tsx", ".js", ".jsx")):
                import_hint = import_hint.rsplit(".", 1)[0]
        else:
            return None
        system_prompt = (
            "Generate a minimal regression test that reproduces the concrete bug described in the review comment. "
            "Return strict JSON with keys path and code. The test should fail on the original code and pass after the fix."
        )
        prompt = (
            f"Language: {self._markdown_language(filepath) or ext}\n"
            f"Framework: {framework}\n"
            f"Default test path: {default_path}\n"
            f"Import hint: {import_hint}\n"
            f"Review comment: {candidate.get('comment', '')}\n"
            f"Function signature: {candidate.get('function_signature', '')}\n"
            f"Function source:\n```{self._markdown_language(filepath)}\n{candidate.get('function_source', '')}\n```"
        )
        raw = self.client.complete(system_prompt, prompt, max_tokens=500).strip()
        payload = self._parse_json_object(raw)
        code = str(payload.get("code", "")).strip()
        path = str(payload.get("path", default_path)).strip() or default_path
        if not code:
            return None
        return {"path": path, "code": code, "framework": framework, "import_hint": import_hint}

    def _materialize_generated_test(self, temp_dir: str, generated_test: dict | None) -> Path | None:
        if not generated_test:
            return None
        relative_path = generated_test.get("path", "").strip()
        if not relative_path:
            return None
        test_path = Path(temp_dir) / relative_path
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(generated_test.get("code", ""), encoding="utf-8")
        return test_path

    def _run_generated_test(self, temp_dir: str, test_path: Path, generated_test: dict | None) -> dict:
        framework = (generated_test or {}).get("framework", "pytest")
        relative_path = os.path.relpath(test_path, temp_dir)
        if framework == "pytest":
            result = self._run_command(["pytest", "-q", relative_path], cwd=temp_dir, timeout=30)
        else:
            result = self._run_command(["npx", "jest", relative_path, "--runInBand"], cwd=temp_dir, timeout=60)
        result = {
            **result,
            "attempted": True,
            "path": relative_path,
            "framework": framework,
            "error_type": self._classify_error(result, phase="test"),
            **self._extract_test_summary(result, [relative_path]),
        }
        return result

    def _empty_repro_result(self, test_path: Path | None, generated_test: dict | None) -> dict:
        return {
            "attempted": False,
            "generated": bool(generated_test),
            "path": str(test_path or ""),
            "framework": (generated_test or {}).get("framework", ""),
            "bug_reproduced": False,
            "fixed": False,
            "status": "unavailable" if generated_test else "not_generated",
            "before": self._empty_test_result(),
            "after": self._empty_test_result(),
        }

    def _repro_failed(self, result: dict) -> bool:
        return result.get("attempted", False) and not result.get("success", True)

    def _repro_status(self, repro: dict) -> str:
        before = repro.get("before", {})
        after = repro.get("after", {})
        if not repro.get("attempted"):
            return repro.get("status", "not_generated")
        if self._repro_failed(before) and after.get("success", False):
            return "confirmed_fixed"
        if self._repro_failed(before) and after.get("attempted") and not after.get("success", True):
            return "confirmed_unfixed"
        if before.get("success", False):
            return "not_reproduced"
        return "inconclusive"

    def _parse_json_object(self, raw: str) -> dict:
        return parse_llm_json(raw)

    def _ast_diff_is_structural(self, old_source: str, new_source: str, extension: str) -> bool:
        return self._ast_change_type(old_source, new_source, extension) == "logic"

    def _ast_change_type(self, old_source: str, new_source: str, extension: str) -> str:
        if extension == ".py":
            try:
                old_tree = ast.dump(ast.parse(old_source), include_attributes=False)
                new_tree = ast.dump(ast.parse(new_source), include_attributes=False)
                if old_tree == new_tree:
                    return "cosmetic"
                normalized_old = ast.dump(_VerifierNameNormalizer().visit(ast.parse(old_source)), include_attributes=False)
                normalized_new = ast.dump(_VerifierNameNormalizer().visit(ast.parse(new_source)), include_attributes=False)
                return "rename" if normalized_old == normalized_new else "logic"
            except SyntaxError:
                return "logic" if old_source != new_source else "cosmetic"
        language = "typescript" if extension in {".ts", ".tsx"} else "javascript"
        parser = self._parsers.get(language)
        if parser is None:
            return "logic" if old_source != new_source else "cosmetic"
        old_tree = parser.parse(old_source.encode("utf-8"))
        new_tree = parser.parse(new_source.encode("utf-8"))
        old_sig = self._tree_signature(old_tree.root_node, old_source.encode("utf-8"), normalize_identifiers=False)
        new_sig = self._tree_signature(new_tree.root_node, new_source.encode("utf-8"), normalize_identifiers=False)
        if old_sig == new_sig:
            return "cosmetic"
        old_norm = self._tree_signature(old_tree.root_node, old_source.encode("utf-8"), normalize_identifiers=True)
        new_norm = self._tree_signature(new_tree.root_node, new_source.encode("utf-8"), normalize_identifiers=True)
        return "rename" if old_norm == new_norm else "logic"

    def _build_parsers(self) -> dict:
        if Parser is None or Language is None:
            return {}
        parsers = {}
        if tsjavascript is not None:
            parser = Parser()
            parser.language = Language(tsjavascript.language())
            parsers["javascript"] = parser
        if tstypescript is not None:
            parser = Parser()
            parser.language = Language(tstypescript.language_typescript())
            parsers["typescript"] = parser
        return parsers

    def _tree_signature(self, node, source: bytes, normalize_identifiers: bool) -> tuple:
        if not getattr(node, "is_named", True):
            return ()
        token = node.type
        if normalize_identifiers and token in {"identifier", "property_identifier", "type_identifier"}:
            value = "<id>"
        else:
            value = source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
            if token not in {"identifier", "property_identifier", "type_identifier", "string", "number"}:
                value = ""
        children = tuple(
            child_sig
            for child in node.children
            for child_sig in [self._tree_signature(child, source, normalize_identifiers)]
            if child_sig
        )
        return (token, value, children)

    def _replace_lines(self, file_source: str, start: int, end: int, replacement: str) -> str:
        lines = file_source.splitlines()
        replacement_lines = replacement.rstrip("\n").splitlines()
        updated = lines[: start - 1] + replacement_lines + lines[end:]
        trailing_newline = "\n" if file_source.endswith("\n") else ""
        return "\n".join(updated) + trailing_newline

    def _generate_unified_diff(self, old_source: str, new_source: str, filepath: str) -> str:
        diff = difflib.unified_diff(
            old_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=filepath,
            tofile=filepath,
        )
        return "".join(diff)

    def _failed_result(
        self,
        candidate: dict,
        reason: str,
        status: str = "inconclusive",
        ast_change_type: str = "unknown",
        mode: str = "full",
    ) -> dict:
        return {
            **candidate,
            "validation": {
                "compiles": False,
                "tests_pass": False,
                "compile_attempted": False,
                "tests_attempted": False,
                "test_results": [],
                "test_pass_ratio": None,
                "compile_error": "",
                "test_error": "",
                "compile_error_type": "",
                "test_error_type": "",
                "compile": self._empty_command_result(),
                "tests": self._empty_test_result(),
                "repro": self._empty_repro_result(None, None),
            },
            "generation": {"attempts": 0, "retried_for": "", "last_error_type": "", "last_error": ""},
            "verification_key": "",
            "verification_reason": reason,
            "verification": {
                "status": status,
                "mode": mode,
                "ast_change_type": ast_change_type,
                "reason": reason,
                "candidate_key": self._candidate_cache_key(candidate),
            },
        }

    def _verification_reason(self, validation: dict, ast_change_type: str, status: str, mode: str) -> str:
        if status == "skipped":
            return "Skipped expensive verification because the draft comment scored below the routing threshold."
        if status == "disproved":
            if ast_change_type == "cosmetic":
                return "Fix was cosmetic only; AST was unchanged."
            if ast_change_type == "rename":
                return "Fix only renamed identifiers and did not change behavior."
        compile_error_type = validation.get("compile_error_type") or validation.get("compile", {}).get("error_type", "")
        if not validation.get("compiles", True):
            error = validation.get("compile_error", "")
            prefix = f"Fix did not compile cleanly ({compile_error_type})." if compile_error_type else "Fix did not compile cleanly."
            return f"{prefix} {error}".strip()
        repro = validation.get("repro", {})
        repro_status = repro.get("status", "")
        if mode == "full" and repro_status == "confirmed_fixed":
            if validation.get("tests_attempted"):
                ratio = validation.get("test_pass_ratio")
                ratio_text = f" Existing tests pass_ratio={ratio:.2f}." if isinstance(ratio, float) else ""
                return f"Generated repro failed before the fix and passed after it.{ratio_text}".strip()
            return "Generated repro failed before the fix and passed after it."
        if validation.get("tests_attempted") and not validation.get("tests_pass", True):
            error = validation.get("test_error", "")
            ratio = validation.get("test_pass_ratio")
            ratio_text = f" pass_ratio={ratio:.2f}." if isinstance(ratio, float) else ""
            return f"Fix failed targeted tests.{ratio_text} {error}".strip()
        if status == "passed":
            if validation.get("tests_attempted"):
                ratio = validation.get("test_pass_ratio")
                ratio_text = f" (pass_ratio={ratio:.2f})" if isinstance(ratio, float) else ""
                return f"Fix compiled, passed targeted tests{ratio_text}, and changed logic structurally."
            return "Fix compiled and changed logic structurally."
        if repro_status == "not_reproduced":
            return "Generated repro test did not reproduce the reported bug."
        if repro_status == "confirmed_unfixed":
            return "Generated repro still fails after the synthesized fix."
        return "Verification was inconclusive."

    def _verification_status(self, validation: dict, ast_change_type: str, mode: str) -> str:
        if ast_change_type in {"cosmetic", "rename"}:
            return "disproved"
        if mode == "skip":
            return "skipped"
        if not validation.get("compiles", True):
            return "inconclusive"
        if validation.get("tests_attempted") and not validation.get("tests_pass", True):
            return "inconclusive"
        if mode == "compile":
            return "inconclusive"
        repro_status = validation.get("repro", {}).get("status", "")
        if validation.get("repro", {}).get("generated"):
            if repro_status == "confirmed_fixed":
                return "passed"
            return "inconclusive"
        if ast_change_type == "logic" and repro_status == "confirmed_fixed":
            return "passed"
        if ast_change_type == "logic" and validation.get("tests_attempted"):
            return "passed"
        if ast_change_type == "logic" and repro_status in {"not_generated", "unavailable"}:
            return "inconclusive"
        return "inconclusive"

    def _empty_command_result(self) -> dict:
        return {
            "attempted": False,
            "success": False,
            "command": "",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "",
            "error_excerpt": "",
            "timed_out": False,
            "error_type": "",
        }

    def _empty_test_result(self) -> dict:
        return {
            **self._empty_command_result(),
            "executed": [],
            "passed_count": 0,
            "failed_count": 0,
            "error_count": 0,
            "total_count": 0,
            "pass_ratio": None,
        }

    def _extract_test_summary(self, result: dict, executed_tests: list[str]) -> dict:
        output = "\n".join(
            part.strip()
            for part in (result.get("stdout", ""), result.get("stderr", ""))
            if part and part.strip()
        )
        passed_count = self._extract_count(output, "passed")
        failed_count = self._extract_count(output, "failed")
        error_count = self._extract_count(output, "error")
        total_count = self._extract_count(output, "total")
        if total_count == 0:
            total_count = passed_count + failed_count + error_count
        if total_count == 0 and result.get("success") and executed_tests:
            total_count = len(executed_tests)
            passed_count = total_count
        pass_ratio = round(passed_count / total_count, 3) if total_count else None
        return {
            "passed_count": passed_count,
            "failed_count": failed_count,
            "error_count": error_count,
            "total_count": total_count,
            "pass_ratio": pass_ratio,
        }

    def _extract_count(self, text: str, label: str) -> int:
        patterns = {
            "passed": r"(\d+)\s+passed\b",
            "failed": r"(\d+)\s+failed\b",
            "error": r"(\d+)\s+errors?\b",
            "total": r"(\d+)\s+total\b",
        }
        pattern = patterns.get(label, "")
        if not pattern:
            return 0
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return sum(int(match) for match in matches)

    def _truncate_output(self, text: str, limit: int = 240) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."

    def _candidate_cache_key(self, candidate: dict) -> str:
        raw = "|".join(
            [
                candidate.get("node_id", ""),
                candidate.get("file", ""),
                str(candidate.get("function_line_start", candidate.get("line_start", 0))),
                str(candidate.get("function_line_end", candidate.get("line_end", 0))),
                candidate.get("comment", ""),
                candidate.get("function_source", ""),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _verification_key(self, comment: str, suggested_code: str) -> str:
        raw = f"{comment}|{suggested_code}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _bundle_file_sources(self, candidates: list[dict], original_source: dict) -> tuple[dict[str, str], list[dict]]:
        by_file: dict[str, list[dict]] = {}
        for candidate in candidates:
            by_file.setdefault(candidate.get("file", ""), []).append(candidate)
        updated_sources: dict[str, str] = {}
        conflicts: list[dict] = []
        for filepath, file_candidates in by_file.items():
            source = original_source.get(filepath)
            if source is None:
                conflicts.append({"file": filepath, "candidates": file_candidates, "reason": "Missing source"})
                continue
            file_conflicts: list[dict] = []
            ordered = sorted(
                file_candidates,
                key=lambda item: (
                    int(item.get("function_line_start", item.get("line_start", 0))),
                    int(item.get("function_line_end", item.get("line_end", 0))),
                ),
            )
            for current, nxt in zip(ordered, ordered[1:]):
                current_end = int(current.get("function_line_end", current.get("line_end", 0)))
                next_start = int(nxt.get("function_line_start", nxt.get("line_start", 0)))
                if current_end >= next_start:
                    file_conflicts.append({"file": filepath, "candidates": [current, nxt], "reason": "Overlapping edits"})
            if file_conflicts:
                conflicts.extend(file_conflicts)
                continue
            updated = source
            for candidate in sorted(
                file_candidates,
                key=lambda item: int(item.get("function_line_start", item.get("line_start", 0))),
                reverse=True,
            ):
                suggested_code = candidate.get("suggested_code", "").rstrip()
                if not suggested_code:
                    conflicts.append({"file": filepath, "candidates": [candidate], "reason": "Missing suggested code"})
                    break
                updated = self._replace_lines(
                    updated,
                    int(candidate.get("function_line_start", candidate.get("line_start", 0))),
                    int(candidate.get("function_line_end", candidate.get("line_end", 0))),
                    suggested_code,
                )
            updated_sources[filepath] = updated
        return updated_sources, conflicts

    def _bundle_test_files(self, candidates: list[dict]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            for path in candidate.get("test_files", []):
                if path in seen:
                    continue
                seen.add(path)
                ordered.append(path)
        return ordered

    def _bundle_reason(self, validation: dict) -> str:
        if validation.get("compile_error_type") == "BundleConflict":
            return validation.get("compile_error", "Combined patch has overlapping edits.")
        if not validation.get("compiles", True):
            error_type = validation.get("compile_error_type", "")
            detail = validation.get("compile_error", "")
            prefix = f"Combined patch did not compile cleanly ({error_type})." if error_type else "Combined patch did not compile cleanly."
            return f"{prefix} {detail}".strip()
        if validation.get("tests_attempted") and not validation.get("tests_pass", True):
            detail = validation.get("test_error", "")
            ratio = validation.get("test_pass_ratio")
            ratio_text = f" pass_ratio={ratio:.2f}." if isinstance(ratio, float) else ""
            return f"Combined patch failed related tests.{ratio_text} {detail}".strip()
        return "Combined patch compiled and passed related tests."

    def _choose_bundle_drop_index(self, candidates: list[dict], validation: dict) -> int:
        conflict_keys = set(validation.get("conflicting_candidate_keys", []))
        if conflict_keys:
            candidate_pool = [
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if (candidate.get("verification_key") or candidate.get("verification", {}).get("candidate_key", "")) in conflict_keys
            ]
            if candidate_pool:
                return min(candidate_pool, key=lambda item: self._bundle_drop_priority(item[1]))[0]
        return min(range(len(candidates)), key=lambda index: self._bundle_drop_priority(candidates[index]))

    def _bundle_drop_priority(self, candidate: dict) -> tuple:
        severity_rank = {"critical": 2, "warning": 1, "suggestion": 0}
        severity = severity_rank.get(str(candidate.get("severity", "warning")).lower(), 1)
        return (
            float(candidate.get("critic_score", candidate.get("confidence", 0.0))),
            severity,
            int(bool(candidate.get("confirmed_by_static", False))),
            int(bool(candidate.get("validation", {}).get("tests_attempted", False))),
        )

    def _annotate_bundle_result(
        self,
        candidate: dict,
        status: str,
        reason: str,
        group_size: int,
        details: dict | None = None,
    ) -> dict:
        annotated = dict(candidate)
        annotated["bundle_verification"] = {
            "status": status,
            "reason": reason,
            "group_size": group_size,
        }
        if details is not None:
            annotated["bundle_verification"]["details"] = {
                "compile_error_type": details.get("compile_error_type", ""),
                "test_error_type": details.get("test_error_type", ""),
                "test_pass_ratio": details.get("test_pass_ratio"),
                "status": details.get("status", ""),
            }
        return annotated

    def _requested_mode(self, verification_modes: list[str] | None, index: int) -> str:
        if verification_modes is None or index >= len(verification_modes):
            return "full"
        mode = str(verification_modes[index] or "full").lower()
        if mode not in {"skip", "compile", "full"}:
            return "full"
        return mode

    def _mode_rank(self, mode: str) -> int:
        return {"skip": 0, "compile": 1, "full": 2}.get(mode, 2)

    def _can_reuse_cached(self, cached: dict, requested_mode: str) -> bool:
        cached_mode = cached.get("verification", {}).get("mode", "full")
        return self._mode_rank(cached_mode) >= self._mode_rank(requested_mode)

    def _skipped_result(self, candidate: dict) -> dict:
        return self._failed_result(
            candidate,
            reason="Skipped expensive verification because the draft comment scored below the routing threshold.",
            status="skipped",
            mode="skip",
        )


class _VerifierNameNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = "__name__"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "__arg__"
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = "__attr__"
        return node
