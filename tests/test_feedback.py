from __future__ import annotations

import json

from ares.feedback import FeedbackCollector, FeedbackLearner, ReviewStrategy


def test_strategy_round_trips_to_disk(tmp_path):
    strategy = ReviewStrategy(
        bug_fix_freq_critical=4,
        change_freq_utility=2,
        utility_blast_radius_threshold=7,
        extra_reviewer_instructions=["Focus on null checks."],
        learned_nit_patterns=["consider extra logging"],
        learned_good_patterns=["missing null check"],
        severity_weights={"critical": 1.2, "warning": 0.8, "suggestion": 0.2},
        min_confidence=0.35,
        version=3,
    )

    strategy.save(str(tmp_path))
    loaded = ReviewStrategy.load(str(tmp_path))

    assert loaded.bug_fix_freq_critical == 4
    assert loaded.change_freq_utility == 2
    assert loaded.utility_blast_radius_threshold == 7
    assert loaded.extra_reviewer_instructions == ["Focus on null checks."]
    assert loaded.learned_nit_patterns == ["consider extra logging"]
    assert loaded.learned_good_patterns == ["missing null check"]
    assert loaded.severity_weights["critical"] == 1.2
    assert loaded.min_confidence == 0.35
    assert loaded.version == 3


def test_collector_records_and_resolves_feedback(tmp_path):
    class FakeGitHub:
        def get_pending_comment_feedback(self, repo_name: str, pr_number: int, pending_comments: list[dict]) -> list[dict]:
            assert repo_name == "owner/repo"
            assert pr_number == 42
            return [
                {
                    **pending_comments[0],
                    "outcome": "addressed",
                    "feedback_label": "upvote",
                    "pr_state": "open",
                    "merged": False,
                }
            ]

    class FakePinecone:
        def __init__(self) -> None:
            self.items: list[dict] = []

        def upsert_feedback(self, items: list[dict]) -> None:
            self.items.extend(items)

    collector = FeedbackCollector(
        str(tmp_path),
        github_client=FakeGitHub(),
        pinecone_client=FakePinecone(),
    )
    records = collector.record_posted_comments(
        "owner/repo",
        42,
        [
            {
                "file": "app.py",
                "line_start": 10,
                "line_end": 10,
                "comment": "Add a null guard before dereferencing payload.",
                "severity": "warning",
                "source": "llm",
                "risk": "critical",
                "pr_intent": "bugfix",
            }
        ],
    )

    summary = collector.collect_feedback()
    outcomes = json.loads(collector.outcomes_path.read_text(encoding="utf-8"))
    remaining = json.loads(collector.pending_path.read_text(encoding="utf-8"))

    assert len(records) == 1
    assert summary["resolved"] == 1
    assert summary["pending"] == 0
    assert outcomes[0]["status"] == "addressed"
    assert remaining == []
    assert collector.pinecone.items[0]["label"] == "upvote"


def test_learner_updates_strategy_from_feedback_outcomes(tmp_path):
    outcomes_path = ReviewStrategy.storage_dir(str(tmp_path)) / "feedback_outcomes.json"
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "status": "ignored",
                    "severity": "suggestion",
                    "pr_intent": "refactor",
                    "risk": "utility",
                    "source": "llm",
                    "comment": "Consider renaming this variable for clarity.",
                },
                {
                    "id": "2",
                    "status": "addressed",
                    "severity": "critical",
                    "pr_intent": "bugfix",
                    "risk": "critical",
                    "source": "llm",
                    "comment": "Add a null check before dereferencing payload.",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    class FakeClient:
        available = True

        def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
            return json.dumps(
                {
                    "extra_reviewer_instructions": ["Focus on null checks in hot paths."],
                    "learned_nit_patterns": ["consider renaming"],
                    "learned_good_patterns": ["null check before dereferencing"],
                    "min_confidence": 0.3,
                    "severity_weights": {"critical": 1.3, "warning": 0.75, "suggestion": 0.2},
                }
            )

    learner = FeedbackLearner(str(tmp_path), strategy=ReviewStrategy(), api_key="")
    learner.client = FakeClient()

    result = learner.improve()
    loaded = ReviewStrategy.load(str(tmp_path))

    assert result["updated"] is True
    assert loaded.extra_reviewer_instructions == ["Focus on null checks in hot paths."]
    assert loaded.learned_nit_patterns == ["consider renaming"]
    assert loaded.learned_good_patterns == ["null check before dereferencing"]
    assert loaded.min_confidence == 0.3
    assert loaded.version == 2
