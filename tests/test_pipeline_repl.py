from __future__ import annotations

from ares.config import AresConfig
from ares.pipeline import Pipeline


def test_pipeline_review_repl_stops_when_scores_plateau(tmp_path):
    pipeline = Pipeline(AresConfig(repo_path=str(tmp_path), workspace_root=str(tmp_path), max_review_passes=3))

    class FakeReviewer:
        def __init__(self) -> None:
            self.review_calls = 0
            self.refine_calls = 0

        def review(self, targets: list[dict]) -> list[dict]:
            self.review_calls += 1
            if self.review_calls == 1:
                return [
                    {
                        "node_id": "app.py::validate",
                        "file": "app.py",
                        "line_start": 10,
                        "line_end": 10,
                        "comment": "Weak first-pass comment.",
                        "confidence": 0.4,
                        "severity": "warning",
                    }
                ]
            return [
                {
                    "node_id": "app.py::validate",
                    "file": "app.py",
                    "line_start": 10,
                    "line_end": 10,
                    "comment": "Worse second-pass comment.",
                    "confidence": 0.3,
                    "severity": "warning",
                }
            ]

        def refine(self, targets: list[dict], candidates: list[dict], scores: list[dict]) -> list[dict]:
            self.refine_calls += 1
            return [{**targets[0], "review_history": [{"attempt": candidates, "critique": scores}]}]

    class FakeCritic:
        def __init__(self) -> None:
            self.calls = 0
            self.prescore_calls = 0

        def prescore_comments(self, comments: list[dict], pr_intent: str) -> list[dict]:
            self.prescore_calls += 1
            if self.prescore_calls == 1:
                return [{"node_id": comments[0]["node_id"], "comment": comments[0]["comment"], "score": 0.9, "keep": True, "reason": "Strong draft."}]
            return [{"node_id": comments[0]["node_id"], "comment": comments[0]["comment"], "score": 0.9, "keep": True, "reason": "Strong draft."}]

        def score_comments(self, comments: list[dict], pr_intent: str, **kwargs) -> list[dict]:
            self.calls += 1
            # Early critique now uses prescore_comments, so score_comments is
            # only called post-verify: call 1 = pass 1, call 2 = pass 2.
            if self.calls <= 1:
                return [
                    {
                        "node_id": "app.py::validate",
                        "comment": comments[0]["comment"],
                        "score": 0.72,
                        "keep": True,
                        "reason": "Concrete enough.",
                    }
                ]
            return [
                {
                    "node_id": "app.py::validate",
                    "comment": comments[0]["comment"],
                    "score": 0.69,
                    "keep": True,
                    "reason": "No improvement.",
                }
            ]

        def average_score(self, scores: list[dict]) -> float:
            return sum(score["score"] for score in scores) / len(scores)

        def all_scores_high(self, scores: list[dict]) -> bool:
            return False

        def select_comments(self, comments: list[dict], scores: list[dict], min_score: float | None = None) -> list[dict]:
            return [comment for comment in comments if comment.get("verification", {}).get("status") == "passed"]

    class FakeVerifier:
        def __init__(self) -> None:
            self.calls = 0
            self.cache_hits = 0
            self.modes: list[str] = []

        def verify_candidates(
            self,
            candidates: list[dict],
            original_source: dict,
            cache: dict[str, dict] | None = None,
            verification_modes: list[str] | None = None,
        ) -> list[dict]:
            cache = cache if cache is not None else {}
            verified = []
            verification_modes = verification_modes or ["full"] * len(candidates)
            for candidate, mode in zip(candidates, verification_modes):
                self.modes.append(mode)
                key = candidate["comment"]
                if key in cache:
                    self.cache_hits += 1
                    verified.append(dict(cache[key]))
                    continue
                self.calls += 1
                result = {
                    **candidate,
                    "validation": {"compiles": True, "tests_pass": True},
                    "verification_reason": "Fix compiled, passed tests, and changed logic structurally.",
                    "verification": {"status": "passed", "ast_change_type": "logic", "mode": mode},
                    "suggested_code": "if payload is None:\n    return ''",
                    "verification_key": key,
                }
                cache[key] = dict(result)
                verified.append(result)
            return verified

    pipeline.reviewer = FakeReviewer()
    pipeline.critic = FakeCritic()
    pipeline.verifier = FakeVerifier()
    targets = [
        {
            "node_id": "app.py::validate",
            "file": "app.py",
            "line_start": 10,
            "line_end": 14,
            "function_source": "def validate(payload):\n    return payload.strip()\n",
            "function_signature": "def validate(payload)",
            "pr_intent": "bugfix in auth (behavior change expected)",
            "review_instruction": "Flag concrete bugs only.",
            "diff_hunk": "@@ -10,2 +10,3 @@",
            "test_files": [],
            "risk": "critical",
            "original_risk": "critical",
            "caller_count": 6,
            "change_type": "logic",
            "context": {
                "callers": "None",
                "callees": "None",
                "tests": "None",
                "bug_history": "2 bug-fix commits",
                "co_changes": [],
            },
        }
    ]

    candidates, scores, metrics = pipeline._run_review_repl(
        targets,
        "bugfix in auth (behavior change expected)",
        {"app.py": "def validate(payload):\n    return payload.strip()\n"},
    )

    assert pipeline.reviewer.review_calls == 2
    assert pipeline.reviewer.refine_calls == 1
    assert pipeline.verifier.calls == 2
    assert pipeline.verifier.modes == ["full", "full"]
    assert candidates[0]["comment"] == "Weak first-pass comment."
    assert scores[0]["score"] == 0.72
    assert len(metrics) == 2
    assert metrics[0]["verified_count"] == 1
    assert metrics[0]["average_score"] > metrics[1]["average_score"]


def test_pipeline_review_repl_skips_reverification_for_unchanged_candidates(tmp_path):
    pipeline = Pipeline(AresConfig(repo_path=str(tmp_path), workspace_root=str(tmp_path), max_review_passes=3))

    class FakeReviewer:
        def __init__(self) -> None:
            self.review_calls = 0

        def review(self, targets: list[dict]) -> list[dict]:
            self.review_calls += 1
            return [
                {
                    "node_id": "app.py::validate",
                    "file": "app.py",
                    "line_start": 10,
                    "line_end": 10,
                    "comment": "Stable verified comment.",
                    "confidence": 0.6,
                    "severity": "warning",
                }
            ]

        def refine(self, targets: list[dict], candidates: list[dict], scores: list[dict]) -> list[dict]:
            return targets

    class FakeCritic:
        def __init__(self) -> None:
            self.calls = 0
            self.prescore_calls = 0

        def prescore_comments(self, comments: list[dict], pr_intent: str) -> list[dict]:
            self.prescore_calls += 1
            return [{"node_id": comments[0]["node_id"], "comment": comments[0]["comment"], "score": 0.9, "keep": True, "reason": "Strong draft."}]

        def score_comments(self, comments: list[dict], pr_intent: str, **kwargs) -> list[dict]:
            self.calls += 1
            if self.calls == 1:
                return [{"node_id": comments[0]["node_id"], "comment": comments[0]["comment"], "score": 0.72, "keep": True, "reason": "Good"}]
            return [{"node_id": comments[0]["node_id"], "comment": comments[0]["comment"], "score": 0.69, "keep": True, "reason": "Plateau"}]

        def average_score(self, scores: list[dict]) -> float:
            return sum(score["score"] for score in scores) / len(scores)

        def all_scores_high(self, scores: list[dict]) -> bool:
            return False

        def select_comments(self, comments: list[dict], scores: list[dict], min_score: float | None = None) -> list[dict]:
            return comments

    class FakeVerifier:
        def __init__(self) -> None:
            self.calls = 0
            self.cache_hits = 0

        def verify_candidates(
            self,
            candidates: list[dict],
            original_source: dict,
            cache: dict[str, dict] | None = None,
            verification_modes: list[str] | None = None,
        ) -> list[dict]:
            cache = cache if cache is not None else {}
            verified = []
            verification_modes = verification_modes or ["full"] * len(candidates)
            for candidate, mode in zip(candidates, verification_modes):
                key = candidate["comment"]
                if key in cache:
                    self.cache_hits += 1
                    verified.append(dict(cache[key]))
                    continue
                self.calls += 1
                result = {
                    **candidate,
                    "validation": {"compiles": True, "tests_pass": True},
                    "verification_reason": "Fix compiled, passed tests, and changed logic structurally.",
                    "verification": {"status": "passed", "ast_change_type": "logic", "mode": mode},
                    "suggested_code": "return payload or ''",
                    "verification_key": key,
                }
                cache[key] = dict(result)
                verified.append(result)
            return verified

    pipeline.reviewer = FakeReviewer()
    pipeline.critic = FakeCritic()
    pipeline.verifier = FakeVerifier()
    targets = [
        {
            "node_id": "app.py::validate",
            "file": "app.py",
            "line_start": 10,
            "line_end": 14,
            "function_source": "def validate(payload):\n    return payload.strip()\n",
            "function_signature": "def validate(payload)",
            "pr_intent": "bugfix in auth (behavior change expected)",
            "review_instruction": "Flag concrete bugs only.",
            "diff_hunk": "@@ -10,2 +10,3 @@",
            "test_files": [],
            "risk": "critical",
            "original_risk": "critical",
            "caller_count": 6,
            "change_type": "logic",
            "context": {
                "callers": "None",
                "callees": "None",
                "tests": "None",
                "bug_history": "2 bug-fix commits",
                "co_changes": [],
            },
        }
    ]

    pipeline._run_review_repl(
        targets,
        "bugfix in auth (behavior change expected)",
        {"app.py": "def validate(payload):\n    return payload.strip()\n"},
    )

    assert pipeline.verifier.calls == 1
    assert pipeline.verifier.cache_hits == 1


def test_pipeline_routes_verification_modes_from_prescores(tmp_path):
    pipeline = Pipeline(AresConfig(repo_path=str(tmp_path), workspace_root=str(tmp_path), max_review_passes=1))

    class FakeReviewer:
        def review(self, targets: list[dict]) -> list[dict]:
            return [
                {"node_id": "a", "file": "a.py", "line_start": 1, "line_end": 1, "comment": "low", "confidence": 0.2, "severity": "warning"},
                {"node_id": "b", "file": "b.py", "line_start": 1, "line_end": 1, "comment": "mid", "confidence": 0.7, "severity": "warning"},
                {"node_id": "c", "file": "c.py", "line_start": 1, "line_end": 1, "comment": "high", "confidence": 0.95, "severity": "critical"},
            ]

        def refine(self, targets: list[dict], candidates: list[dict], scores: list[dict]) -> list[dict]:
            return targets

    class FakeCritic:
        _prescore_map = {"a": 0.2, "b": 0.5, "c": 0.85}

        def prescore_comments(self, comments: list[dict], pr_intent: str) -> list[dict]:
            return [
                {"node_id": c["node_id"], "comment": c["comment"], "score": self._prescore_map.get(c["node_id"], 0.5), "keep": self._prescore_map.get(c["node_id"], 0.5) >= 0.6, "reason": "prescore"}
                for c in comments
            ]

        def score_comments(self, comments: list[dict], pr_intent: str, **kwargs) -> list[dict]:
            return [
                {"node_id": comment["node_id"], "comment": comment["comment"], "score": 0.9, "keep": True, "reason": "final"}
                for comment in comments
            ]

        def average_score(self, scores: list[dict]) -> float:
            return sum(score["score"] for score in scores) / len(scores)

        def all_scores_high(self, scores: list[dict]) -> bool:
            return False

        def select_comments(self, comments: list[dict], scores: list[dict], min_score: float | None = None) -> list[dict]:
            return comments

    class FakeVerifier:
        def __init__(self) -> None:
            self.modes: list[str] = []

        def verify_candidates(
            self,
            candidates: list[dict],
            original_source: dict,
            cache: dict[str, dict] | None = None,
            verification_modes: list[str] | None = None,
        ) -> list[dict]:
            self.modes = list(verification_modes or [])
            results = []
            for candidate, mode in zip(candidates, verification_modes or []):
                status = "skipped" if mode == "skip" else "passed"
                results.append(
                    {
                        **candidate,
                        "validation": {"compiles": True, "tests_pass": True},
                        "verification_reason": mode,
                        "verification": {"status": status, "ast_change_type": "logic", "mode": mode},
                    }
                )
            return results

    pipeline.reviewer = FakeReviewer()
    pipeline.critic = FakeCritic()
    pipeline.verifier = FakeVerifier()

    pipeline._run_review_repl(
        [{"node_id": "seed", "file": "app.py", "line_start": 1, "line_end": 1, "function_source": "def x():\n    pass\n", "function_signature": "def x()", "pr_intent": "bugfix", "review_instruction": "review", "diff_hunk": "@@", "test_files": [], "risk": "critical", "original_risk": "critical", "caller_count": 1, "change_type": "logic", "context": {"callers": "", "callees": "", "tests": "", "bug_history": "", "co_changes": []}}],
        "bugfix",
        {"app.py": "def x():\n    pass\n"},
    )

    assert pipeline.verifier.modes == ["skip", "compile", "full"]


def test_pipeline_bundle_verification_drops_failed_llm_comment(tmp_path):
    pipeline = Pipeline(AresConfig(repo_path=str(tmp_path), workspace_root=str(tmp_path), max_review_passes=1))

    class FakeVerifier:
        def verify_bundle(self, candidates: list[dict], original_source: dict) -> dict:
            return {
                "survivors": [
                    {
                        **candidates[0],
                        "bundle_verification": {"status": "passed", "reason": "bundle ok", "group_size": 2},
                    }
                ],
                "dropped": [
                    {
                        **candidates[1],
                        "bundle_verification": {"status": "failed", "reason": "bundle conflict", "group_size": 2},
                    }
                ],
                "group_size": 2,
            }

    pipeline.verifier = FakeVerifier()
    comments = [
        {
            "source": "llm",
            "node_id": "app.py::a",
            "file": "app.py",
            "comment": "keep",
            "suggested_code": "def a():\n    return 1",
            "validation": {"compiles": True},
            "verification_key": "keep",
        },
        {
            "source": "llm",
            "node_id": "app.py::b",
            "file": "app.py",
            "comment": "drop",
            "suggested_code": "def b():\n    return 2",
            "validation": {"compiles": True},
            "verification_key": "drop",
        },
        {
            "source": "static",
            "tool": "ruff",
            "file": "app.py",
            "line_start": 1,
            "line_end": 1,
            "message": "static finding",
        },
    ]

    filtered = pipeline._apply_bundle_verification(comments, {"app.py": "def a():\n    return 0\n\ndef b():\n    return 0\n"})

    assert len(filtered) == 2
    assert {comment.get("verification_key", comment.get("source")) for comment in filtered} == {"keep", "static"}
    static_comment = next(comment for comment in filtered if comment.get("source") == "static")
    assert static_comment["bundle_verification"]["status"] == "skipped"


def test_pipeline_attaches_review_loop_metadata(tmp_path):
    pipeline = Pipeline(AresConfig(repo_path=str(tmp_path), workspace_root=str(tmp_path), max_review_passes=2))
    pipeline.last_review_loop = {"passes": [{"pass": 1, "average_score": 0.8}], "best_average_score": 0.8}
    comments = [{"node_id": "app.py::validate", "comment": "Null dereference risk."}]
    scores = [
        {
            "node_id": "app.py::validate",
            "comment": "Null dereference risk.",
            "score": 0.81,
            "reason": "Concrete runtime failure.",
        }
    ]

    pipeline._attach_review_loop_metadata(comments, pipeline.last_review_loop["passes"], scores)

    assert comments[0]["review_loop"]["best_average_score"] == 0.8
    assert comments[0]["critic_score"] == 0.81
    assert comments[0]["critique_reason"] == "Concrete runtime failure."
