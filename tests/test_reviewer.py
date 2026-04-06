from __future__ import annotations

from ares.agents.reviewer import Reviewer


def test_reviewer_appends_strategy_instructions_to_system_prompt():
    reviewer = Reviewer(
        api_key="",
        extra_instructions=["Focus on null checks.", "Skip speculative error-handling advice."],
    )

    prompt = reviewer._system_prompt()

    assert "Focus on null checks." in prompt
    assert "Skip speculative error-handling advice." in prompt


def test_reviewer_refine_adds_history_to_future_prompt():
    reviewer = Reviewer(api_key="")
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
    candidates = [
        {
            "node_id": "app.py::validate",
            "comment": "Consider renaming payload for readability.",
            "severity": "suggestion",
            "confidence": 0.4,
            "reasoning": "payload is vague",
        }
    ]
    scores = [
        {
            "node_id": "app.py::validate",
            "comment": "Consider renaming payload for readability.",
            "score": 0.2,
            "keep": False,
            "reason": "This is naming feedback, not a concrete bug.",
        }
    ]

    refined = reviewer.refine(targets, candidates, scores)
    prompt = reviewer._build_prompt(refined[0])

    assert "Previous passes and critique" in prompt
    assert "Consider renaming payload for readability." in prompt
    assert "This is naming feedback, not a concrete bug." in prompt
