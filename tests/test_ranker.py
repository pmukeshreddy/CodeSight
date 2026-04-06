from __future__ import annotations

from ares.feedback import ReviewStrategy
from ares.ranker.ranker import Ranker


def test_ranker_deduplicates_and_caps():
    static_findings = [
        {
            "tool": "ruff",
            "file": "app.py",
            "line_start": 10,
            "line_end": 10,
            "message": "undefined name foo",
            "severity": "high",
            "rule_id": "F821",
        },
        {
            "tool": "semgrep",
            "file": "billing.py",
            "line_start": 22,
            "line_end": 24,
            "message": "Possible SQL injection",
            "severity": "high",
            "rule_id": "python.sql-injection",
        },
    ]
    verified_comments = [
        {
            "file": "app.py",
            "line_start": 10,
            "line_end": 10,
            "comment": "This references foo before it is defined and will raise at runtime.",
            "severity": "critical",
            "confidence": 0.92,
            "suggested_code": "bar = foo",
            "verification": {"status": "passed"},
        },
        {
            "file": "auth.py",
            "line_start": 5,
            "line_end": 7,
            "comment": "This skips the authorization check when the token is empty.",
            "severity": "critical",
            "confidence": 0.97,
            "suggested_code": "if not token:\n    raise ValueError('missing token')",
            "verification": {"status": "passed"},
        },
        {
            "file": "misc.py",
            "line_start": 1,
            "line_end": 1,
            "comment": "Minor issue.",
            "severity": "suggestion",
            "confidence": 0.2,
        },
    ]

    ranked = Ranker(max_comments=2).rank_and_cap(static_findings, verified_comments)

    assert len(ranked) == 2
    assert ranked[0]["file"] == "auth.py"
    assert any(comment.get("confirmed_by_static") for comment in ranked if comment["file"] == "app.py")
    assert all("github_body" in comment for comment in ranked)


def test_ranker_applies_strategy_thresholds_and_weights():
    ranker = Ranker(
        max_comments=3,
        strategy=ReviewStrategy(
            min_confidence=0.6,
            severity_weights={"critical": 0.4, "warning": 0.9, "suggestion": 0.1},
        ),
    )

    ranked = ranker.rank_and_cap(
        [],
        [
            {
                "file": "a.py",
                "line_start": 1,
                "line_end": 1,
                "comment": "Low-confidence critical comment.",
                "severity": "critical",
                "confidence": 0.55,
            },
            {
                "file": "b.py",
                "line_start": 2,
                "line_end": 2,
                "comment": "High-confidence warning comment.",
                "severity": "warning",
                "confidence": 0.8,
            },
        ],
    )

    assert len(ranked) == 1
    assert ranked[0]["file"] == "b.py"


def test_ranker_only_emits_suggestions_for_passed_verifications():
    ranker = Ranker(max_comments=2)

    ranked = ranker.rank_and_cap(
        [],
        [
            {
                "file": "a.py",
                "line_start": 1,
                "line_end": 1,
                "comment": "Compile-proven comment.",
                "severity": "warning",
                "confidence": 0.9,
                "suggested_code": "return 1",
                "verification": {"status": "passed"},
            },
            {
                "file": "b.py",
                "line_start": 2,
                "line_end": 2,
                "comment": "Inconclusive verifier comment.",
                "severity": "warning",
                "confidence": 0.95,
                "suggested_code": "return 2",
                "verification": {"status": "inconclusive"},
            },
        ],
    )

    by_file = {comment["file"]: comment for comment in ranked}
    assert "```suggestion" in by_file["a.py"]["github_body"]
    assert "```suggestion" not in by_file["b.py"]["github_body"]
