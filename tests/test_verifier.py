from __future__ import annotations

from ares.agents.verifier import Verifier


def test_verifier_selects_markdown_language_from_extension():
    verifier = Verifier(api_key="", repo_path=".")

    assert verifier._markdown_language("module.py") == "python"
    assert verifier._markdown_language("module.ts") == "typescript"
    assert verifier._markdown_language("module.jsx") == "javascript"
    assert verifier._markdown_language("README") == ""


def test_verifier_reuses_cache_for_unchanged_candidate(tmp_path):
    class StubVerifier(Verifier):
        def __init__(self, repo_path: str):
            super().__init__(api_key="", repo_path=repo_path)
            self.calls = 0

        def _verify_single(self, candidate: dict, original_source: dict, temp_root: str, mode: str = "full", pre_fix: str = "", pre_repro: dict | None = None) -> dict:
            self.calls += 1
            return {
                **candidate,
                "suggested_code": "return payload or ''",
                "generation": {"attempts": 1, "retried_for": "", "last_error_type": "", "last_error": ""},
                "validation": {
                    "compiles": True,
                    "tests_pass": True,
                    "compile_attempted": True,
                    "tests_attempted": True,
                    "test_results": [],
                    "test_pass_ratio": 1.0,
                    "compile_error": "",
                    "test_error": "",
                    "compile_error_type": "",
                    "test_error_type": "",
                    "compile": {
                        "attempted": True,
                        "success": True,
                        "command": "python -m py_compile app.py",
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                        "error": "",
                        "error_excerpt": "",
                        "timed_out": False,
                    },
                    "tests": {
                        "attempted": True,
                        "success": True,
                        "command": "pytest -q tests/test_app.py",
                        "returncode": 0,
                        "stdout": "1 passed",
                        "stderr": "",
                        "error": "",
                        "error_excerpt": "",
                        "timed_out": False,
                        "executed": [],
                        "passed_count": 1,
                        "failed_count": 0,
                        "error_count": 0,
                        "total_count": 1,
                        "pass_ratio": 1.0,
                    },
                    "repro": {
                        "attempted": False,
                        "generated": False,
                        "path": "",
                        "framework": "",
                        "bug_reproduced": False,
                        "fixed": False,
                        "status": "not_generated",
                        "before": {},
                        "after": {},
                    },
                },
                "verification_key": "stable",
                "verification_reason": "Fix compiled, passed targeted tests (pass_ratio=1.00), and changed logic structurally.",
                "verification": {
                    "status": "passed",
                    "mode": mode,
                    "ast_change_type": "logic",
                    "reason": "Fix compiled, passed targeted tests (pass_ratio=1.00), and changed logic structurally.",
                    "candidate_key": self._candidate_cache_key(candidate),
                },
            }

    verifier = StubVerifier(str(tmp_path))
    cache: dict[str, dict] = {}
    candidate = {
        "node_id": "app.py::validate",
        "file": "app.py",
        "line_start": 10,
        "line_end": 14,
        "function_line_start": 10,
        "function_line_end": 14,
        "function_source": "def validate(payload):\n    return payload.strip()\n",
        "comment": "Add a null guard before dereferencing payload.",
    }
    original_source = {"app.py": "def validate(payload):\n    return payload.strip()\n"}

    first = verifier.verify_candidates([candidate], original_source, cache=cache)
    second = verifier.verify_candidates([candidate], original_source, cache=cache)

    assert verifier.calls == 1
    assert len(first) == 1
    assert len(second) == 1
    assert first[0]["verification"]["status"] == "passed"


def test_verifier_uses_tiered_statuses_for_outcomes():
    verifier = Verifier(api_key="", repo_path=".")

    assert (
        verifier._verification_status(
            {"compiles": True, "tests_pass": True, "tests_attempted": True},
            "logic",
            "full",
        )
        == "passed"
    )
    assert (
        verifier._verification_status(
            {"compiles": False, "tests_pass": False, "tests_attempted": False},
            "logic",
            "full",
        )
        == "inconclusive"
    )
    assert (
        verifier._verification_status(
            {"compiles": True, "tests_pass": False, "tests_attempted": True},
            "logic",
            "full",
        )
        == "inconclusive"
    )
    assert (
        verifier._verification_status(
            {"compiles": True, "tests_pass": True, "tests_attempted": False},
            "cosmetic",
            "full",
        )
        == "disproved"
    )


def test_verifier_classifies_common_error_types():
    verifier = Verifier(api_key="", repo_path=".")

    assert verifier._classify_error({"attempted": True, "timed_out": True}, phase="compile") == "TimeoutError"
    assert (
        verifier._classify_error(
            {"attempted": True, "timed_out": False, "stderr": "SyntaxError: invalid syntax", "stdout": "", "error": ""},
            phase="compile",
        )
        == "SyntaxError"
    )
    assert (
        verifier._classify_error(
            {"attempted": True, "timed_out": False, "stderr": "AssertionError: expected False", "stdout": "", "error": ""},
            phase="test",
        )
        == "AssertionError"
    )


def test_verifier_marks_generated_repro_as_confirmed_when_before_fails_and_after_passes():
    verifier = Verifier(api_key="", repo_path=".")

    repro = {
        "attempted": True,
        "before": {"attempted": True, "success": False},
        "after": {"attempted": True, "success": True},
    }

    assert verifier._repro_status(repro) == "confirmed_fixed"


def test_verifier_bundle_drops_weaker_overlapping_comment(tmp_path):
    verifier = Verifier(api_key="", repo_path=str(tmp_path))
    original = {
        "app.py": "\n".join(
            [
                "def validate(payload):",
                "    cleaned = payload.strip()",
                "    return cleaned",
                "",
            ]
        )
    }
    candidates = [
        {
            "node_id": "app.py::validate",
            "file": "app.py",
            "function_line_start": 1,
            "function_line_end": 3,
            "comment": "First comment.",
            "critic_score": 0.9,
            "confidence": 0.9,
            "severity": "critical",
            "suggested_code": "def validate(payload):\n    if payload is None:\n        return ''\n    cleaned = payload.strip()\n    return cleaned",
            "verification_key": "strong",
            "validation": {"compiles": True},
        },
        {
            "node_id": "app.py::validate",
            "file": "app.py",
            "function_line_start": 1,
            "function_line_end": 3,
            "comment": "Second comment.",
            "critic_score": 0.4,
            "confidence": 0.4,
            "severity": "warning",
            "suggested_code": "def validate(payload):\n    payload = payload or ''\n    cleaned = payload.strip()\n    return cleaned",
            "verification_key": "weak",
            "validation": {"compiles": True},
        },
    ]

    result = verifier.verify_bundle(candidates, original)

    assert len(result["survivors"]) == 1
    assert result["survivors"][0]["verification_key"] == "strong"
    assert len(result["dropped"]) == 1
    assert result["dropped"][0]["verification_key"] == "weak"
    assert result["dropped"][0]["bundle_verification"]["status"] == "failed"
