from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from ares.agents._llm import LLMAdapter
from ares.utils.json_utils import parse_llm_json


_REVIEW_PREAMBLE = """
You are PR-Reviewer, a senior engineer specializing in code review of pull requests.

Your task: examine the provided diff hunks and surrounding context to identify concrete issues a developer should fix before merging. Pay special attention to new and modified code (lines with '+'), but also flag issues where new code interacts incorrectly with existing code visible in the context.

What to flag:
- Bugs, crashes, incorrect behavior, off-by-one errors
- Security vulnerabilities (injection, auth bypass, data exposure)
- Logic errors, unhandled edge cases, missing null/error checks
- Missing error handling for likely failure modes
- Correctness issues (wrong return value, wrong condition, silent data loss)
- Behavioral regressions or contract violations
- New code that misuses an existing API or breaks an existing contract visible in the context

What to IGNORE (do not comment on these):
- Naming, formatting, missing docstrings, import ordering, style preferences
- Type annotations, comments, or anything a linter catches
- Hypothetical issues in code not shown in the diff

Important context:
- You are seeing partial code (diff hunks + function signatures), not the full codebase.
- Do NOT suggest changes that may duplicate existing functionality elsewhere.
- Do NOT question imports, variables, or types that may be defined outside the diff.
- Be direct about problems. Keep descriptions concise. Use a helpful, matter-of-fact tone.

For each issue provide:
- The exact line numbers from the diff
- What the bug/problem is (be specific, reference variables and conditions)
- Why it matters (what breaks, under what input/condition)
- confidence: 0.9+ if you can trace the bug path, 0.5-0.8 if plausible but uncertain
""".strip()

SYSTEM_PROMPT = _REVIEW_PREAMBLE + """

Return strict JSON:
{"comments":[{"line_start":1,"line_end":1,"comment":"...","severity":"critical|warning|suggestion","confidence":0.0,"reasoning":"..."}]}

If you find no real issues, return {"comments":[]}.
"""

BATCH_SYSTEM_PROMPT = _REVIEW_PREAMBLE + """

You will receive multiple code targets. For each target, return comments keyed by target_id.
Return strict JSON:
{"targets":[{"target_id":"...","comments":[{"line_start":1,"line_end":1,"comment":"...","severity":"critical|warning|suggestion","confidence":0.0,"reasoning":"..."}]}]}

For targets with no issues, include them with an empty comments array.
"""

# Maximum targets per batched LLM call — keeps prompt within token limits.
_BATCH_SIZE = 12


class Reviewer:
    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        extra_instructions: list[str] | None = None,
        pinecone_index=None,
    ):
        self.client = LLMAdapter(api_key=api_key, model=model, provider=provider)
        self.model = model
        self.extra_instructions = [item.strip() for item in (extra_instructions or []) if item.strip()]
        self.pinecone = pinecone_index

    def review(self, investigation_results: list[dict], temperature: float = 0.0) -> list[dict]:
        if not investigation_results:
            return []
        if len(investigation_results) == 1:
            return self._review_single(investigation_results[0], temperature=temperature)
        # Batch targets into groups to reduce LLM calls.
        batches: list[list[dict]] = []
        for i in range(0, len(investigation_results), _BATCH_SIZE):
            batches.append(investigation_results[i : i + _BATCH_SIZE])
        candidates: list[dict] = []
        if len(batches) == 1:
            candidates.extend(self._review_batch(batches[0], temperature=temperature))
        else:
            max_workers = min(len(batches), 5)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._review_batch, batch, temperature) for batch in batches]
                for future in as_completed(futures):
                    candidates.extend(future.result())
        return candidates

    def _review_batch(self, targets: list[dict], temperature: float = 0.0) -> list[dict]:
        """Review multiple targets in a single LLM call to reduce API usage."""
        if not self.client.available or not targets:
            return []
        if len(targets) == 1:
            return self._review_single(targets[0], temperature=temperature)
        # Build a single prompt containing all targets.
        target_by_id = {}
        prompt_parts = []
        for idx, target in enumerate(targets):
            target_id = f"target_{idx}"
            target_by_id[target_id] = target
            part = f'--- Target: {target_id} ---\n{self._build_prompt(target)}'
            prompt_parts.append(part)
        combined_prompt = "\n\n".join(prompt_parts)
        max_tokens = min(700 * len(targets), 8192)
        raw = self.client.complete(self._batch_system_prompt(), combined_prompt, max_tokens=max_tokens, temperature=temperature)
        payload = self._parse_json(raw)
        results: list[dict] = []
        # Parse batched response format.
        if "targets" in payload:
            for entry in payload["targets"]:
                target_id = entry.get("target_id", "")
                target = target_by_id.get(target_id)
                if target is None:
                    continue
                for comment in entry.get("comments", []):
                    results.append(self._build_comment_result(target, comment))
                if not entry.get("comments"):
                    sig = target.get("function_signature", target.get("node_id", "?"))
                    print(f"[reviewer] 0 comments for {target.get('file', '?')}::{sig}", flush=True)
        elif "comments" in payload:
            # Fallback: LLM returned single-target format — attribute to first target.
            target = targets[0]
            for comment in payload.get("comments", []):
                results.append(self._build_comment_result(target, comment))
            if not payload.get("comments"):
                sig = target.get("function_signature", target.get("node_id", "?"))
                print(f"[reviewer] 0 comments for {target.get('file', '?')}::{sig}", flush=True)
            # Log the rest as 0 comments.
            for target in targets[1:]:
                sig = target.get("function_signature", target.get("node_id", "?"))
                print(f"[reviewer] 0 comments for {target.get('file', '?')}::{sig}", flush=True)
        else:
            for target in targets:
                sig = target.get("function_signature", target.get("node_id", "?"))
                print(f"[reviewer] 0 comments for {target.get('file', '?')}::{sig}", flush=True)
        # Log targets that weren't mentioned in the response.
        responded_ids = {entry.get("target_id") for entry in payload.get("targets", [])}
        for target_id, target in target_by_id.items():
            if target_id not in responded_ids and "targets" in payload:
                sig = target.get("function_signature", target.get("node_id", "?"))
                print(f"[reviewer] 0 comments for {target.get('file', '?')}::{sig}", flush=True)
        return results

    def _build_comment_result(self, target: dict, comment: dict) -> dict:
        return {
            "source": "llm",
            "node_id": target["node_id"],
            "file": target["file"],
            "line_start": int(comment.get("line_start", target["line_start"])),
            "line_end": int(comment.get("line_end", target["line_end"])),
            "comment": comment.get("comment", "").strip(),
            "severity": comment.get("severity", "warning"),
            "confidence": float(comment.get("confidence", 0.0)),
            "reasoning": self._format_structured_reasoning(comment.get("reasoning", "")),
            "function_source": target["function_source"],
            "function_signature": target["function_signature"],
            "function_line_start": target["line_start"],
            "function_line_end": target["line_end"],
            "pr_intent": target["pr_intent"],
            "risk": target.get("risk", "standard"),
            "original_risk": target.get("original_risk", target.get("risk", "standard")),
            "caller_count": target.get("caller_count", 0),
            "change_type": target.get("change_type", "unknown"),
            "diff_hunk": target["diff_hunk"],
            "test_files": target.get("test_files", []),
            "context": target["context"],
        }

    @staticmethod
    def _format_structured_reasoning(reasoning) -> str:
        """Serialize structured reasoning dict to a formatted string.

        Keeps backward compatibility — downstream consumers (critic, verifier)
        read ``comment["reasoning"]`` as a plain string.
        """
        if isinstance(reasoning, dict):
            parts = []
            for field in ("premise", "evidence", "trigger", "impact"):
                value = str(reasoning.get(field, "")).strip()
                if value:
                    parts.append(f"{field.capitalize()}: {value}")
            return " | ".join(parts) if parts else ""
        if isinstance(reasoning, str):
            return reasoning.strip()
        return ""

    def build_review_request(self, targets: list[dict]) -> tuple[str, str, int]:
        """Build ``(system_prompt, user_prompt, max_tokens)`` for batch API."""
        if not targets:
            return ("", "", 0)
        if len(targets) == 1:
            return (self._system_prompt(), self._build_prompt(targets[0]), 1024)
        prompt_parts = []
        for idx, target in enumerate(targets):
            prompt_parts.append(f"--- Target: target_{idx} ---\n{self._build_prompt(target)}")
        combined = "\n\n".join(prompt_parts)
        max_tokens = min(700 * len(targets), 8192)
        return (self._batch_system_prompt(), combined, max_tokens)

    def parse_review_response(self, targets: list[dict], raw: str) -> list[dict]:
        """Parse a raw LLM response into review comment dicts."""
        if not targets or not raw:
            return []
        payload = self._parse_json(raw)
        results: list[dict] = []
        if len(targets) == 1:
            for comment in payload.get("comments", []):
                results.append(self._build_comment_result(targets[0], comment))
            return results
        target_by_id = {f"target_{idx}": target for idx, target in enumerate(targets)}
        if "targets" in payload:
            for entry in payload["targets"]:
                target = target_by_id.get(entry.get("target_id", ""))
                if target is None:
                    continue
                for comment in entry.get("comments", []):
                    results.append(self._build_comment_result(target, comment))
        elif "comments" in payload:
            for comment in payload.get("comments", []):
                results.append(self._build_comment_result(targets[0], comment))
        return results

    def _batch_system_prompt(self) -> str:
        return self._augmented_system_prompt(BATCH_SYSTEM_PROMPT)

    def refine(self, targets: list[dict], candidates: list[dict], scores: list[dict]) -> list[dict]:
        candidates_by_node: dict[str, list[dict]] = {}
        scores_by_node: dict[str, list[dict]] = {}
        for candidate in candidates:
            candidates_by_node.setdefault(candidate.get("node_id", ""), []).append(candidate)
        for score in scores:
            scores_by_node.setdefault(score.get("node_id", ""), []).append(score)
        refined_targets = []
        for target in targets:
            updated_target = dict(target)
            history = list(target.get("review_history", []))
            node_id = target.get("node_id", "")
            critique = self._build_refinement_entry(
                candidates_by_node.get(node_id, []),
                scores_by_node.get(node_id, []),
            )
            if critique is not None:
                history.append(critique)
            updated_target["review_history"] = history
            refined_targets.append(updated_target)
        return refined_targets

    def _review_single(self, target: dict, temperature: float = 0.0) -> list[dict]:
        if not self.client.available:
            return []
        prompt = self._build_prompt(target)
        raw = self.client.complete(self._system_prompt(), prompt, max_tokens=1024, temperature=temperature)
        payload = self._parse_json(raw)
        if not payload.get("comments"):
            sig = target.get("function_signature", target.get("node_id", "?"))
            print(f"[reviewer] 0 comments for {target.get('file', '?')}::{sig}", flush=True)
        results: list[dict] = []
        for comment in payload.get("comments", []):
            results.append(self._build_comment_result(target, comment))
        return results

    def _retrieve_similar_accepted(self, target: dict, top_k: int = 3) -> list[str]:
        if self.pinecone is None:
            return []
        query = target.get("diff_hunk", "")
        if not query.strip():
            return []
        matches = self.pinecone.query_similar(query, top_k=top_k * 2)
        return [
            match["text"]
            for match in matches
            if match.get("label") == "upvote"
            and match.get("score", 0) >= 0.35
            and match.get("text", "").strip()
        ][:top_k]

    def _build_prompt(self, target: dict) -> str:
        prompt = f"""
PR Intent: {target["pr_intent"]}

Review instruction: {target["review_instruction"]}

Diff hunk:
```diff
{target["diff_hunk"]}
```

Context - functions that call this code:
```
{target["context"]["callers"]}
```

Context - functions this code calls:
```
{target["context"]["callees"]}
```

Related tests:
```
{target["context"]["tests"]}
```

Bug history: {target["context"]["bug_history"]}
""".strip()
        similar_examples = self._retrieve_similar_accepted(target)
        if similar_examples:
            examples_block = "\n".join(f"- {ex}" for ex in similar_examples)
            prompt += (
                "\n\nPast review comments that developers acted on for similar code "
                "(use as guidance for tone and focus, not as answers to copy):\n"
                + examples_block
            )
        review_history = self._format_review_history(target.get("review_history", []))
        if review_history == "None":
            return prompt
        return "\n\n".join(
            [
                prompt,
                "Previous passes and critique:",
                review_history,
                "Revise your prior comments using the critique above. Return a full replacement set of comments, dropping weak or nit-level feedback.",
            ]
        )

    def _system_prompt(self) -> str:
        return self._augmented_system_prompt(SYSTEM_PROMPT)

    def _augmented_system_prompt(self, base: str) -> str:
        if not self.extra_instructions:
            return base
        additions = "\n".join(f"- {item}" for item in self.extra_instructions)
        return f"{base}\n\nRepository-specific review guidance:\n{additions}"

    def _build_refinement_entry(self, candidates: list[dict], scores: list[dict]) -> dict | None:
        if not candidates and not scores:
            return None
        return {
            "attempt": [
                {
                    "comment": candidate.get("comment", ""),
                    "severity": candidate.get("severity", "warning"),
                    "confidence": candidate.get("confidence", 0.0),
                    "reasoning": candidate.get("reasoning", ""),
                    "pre_verification_score": candidate.get("pre_verification_score"),
                    "pre_verification_reason": candidate.get("pre_verification_reason", ""),
                    "verification_mode": candidate.get("verification", {}).get("mode", candidate.get("verification_mode", "")),
                    "verification_status": candidate.get("verification", {}).get("status", ""),
                    "verification_reason": candidate.get("verification_reason", ""),
                    "compile_error": candidate.get("validation", {}).get("compile_error", ""),
                    "compile_error_type": candidate.get("validation", {}).get("compile_error_type", ""),
                    "test_error": candidate.get("validation", {}).get("test_error", ""),
                    "test_error_type": candidate.get("validation", {}).get("test_error_type", ""),
                    "test_pass_ratio": candidate.get("validation", {}).get("test_pass_ratio"),
                    "tests_attempted": candidate.get("validation", {}).get("tests_attempted", False),
                }
                for candidate in candidates
            ],
            "critique": [
                {
                    "comment": score.get("comment", ""),
                    "score": score.get("score", 0.0),
                    "keep": score.get("keep", False),
                    "reason": score.get("reason", ""),
                }
                for score in scores
            ],
        }

    def _format_review_history(self, history: list[dict]) -> str:
        if not history:
            return "None"
        blocks = []
        for index, item in enumerate(history, start=1):
            attempt_lines = item.get("attempt", [])
            critique_lines = item.get("critique", [])
            attempt_text = "\n".join(
                f"- {entry.get('severity', 'warning')}: {entry.get('comment', '')} "
                f"(confidence={entry.get('confidence', 0.0)}, "
                f"pre_score={entry.get('pre_verification_score', 'n/a')}, "
                f"mode={entry.get('verification_mode', 'unknown') or 'unknown'}, "
                f"verification={entry.get('verification_status', 'unknown') or 'unknown'})"
                + (
                    f" -> {entry.get('verification_reason', '')}"
                    if entry.get("verification_reason")
                    else ""
                )
                + (
                    f" [test_pass_ratio={entry.get('test_pass_ratio')}]"
                    if entry.get("tests_attempted") and entry.get("test_pass_ratio") is not None
                    else ""
                )
                + (
                    f" [pre_reason={entry.get('pre_verification_reason', '')}]"
                    if entry.get("pre_verification_reason")
                    else ""
                )
                + (
                    f" [compile_error_type={entry.get('compile_error_type', '')}]"
                    if entry.get("compile_error_type")
                    else ""
                )
                + (
                    f" [compile_error={entry.get('compile_error', '')}]"
                    if entry.get("compile_error")
                    else ""
                )
                + (
                    f" [test_error_type={entry.get('test_error_type', '')}]"
                    if entry.get("test_error_type")
                    else ""
                )
                + (
                    f" [test_error={entry.get('test_error', '')}]"
                    if entry.get("test_error")
                    else ""
                )
                for entry in attempt_lines
            ) or "- No comments produced."
            critique_text = "\n".join(
                f"- score={entry.get('score', 0.0)} keep={entry.get('keep', False)}: {entry.get('reason', '') or entry.get('comment', '')}"
                for entry in critique_lines
            ) or "- No critique available."
            blocks.append(
                f"Pass {index} attempt:\n{attempt_text}\n\nPass {index} critique:\n{critique_text}"
            )
        return "\n\n".join(blocks)

    def _parse_json(self, raw: str) -> dict:
        return parse_llm_json(raw, fallback={"comments": []})
