from __future__ import annotations

from ares.agents.critic import Critic
from ares.feedback import ReviewStrategy


def test_seeded_nit_matches_normalized_substring_patterns():
    critic = Critic()

    assert critic._is_seeded_nit("Consider renaming `payload_value` to `payload` for clarity.")
    assert critic._is_seeded_nit("Import-ordering looks off in this module.")


def test_seeded_nit_does_not_require_full_text_similarity():
    critic = Critic()

    assert not critic._is_seeded_nit(
        "This branch skips validation and can accept an expired token."
    )


def test_critic_uses_learned_patterns_from_strategy():
    critic = Critic(
        strategy=ReviewStrategy(
            learned_nit_patterns=["micro optimization"],
            learned_good_patterns=["null check before dereferencing"],
        )
    )

    assert critic._is_seeded_nit("This feels like a micro-optimization in a cold path.")

    assert critic._matches_learned_good_pattern("Missing a null check before dereferencing payload.")


def test_critic_score_comments_returns_reasons_and_keep_flags():
    critic = Critic(
        strategy=ReviewStrategy(
            learned_nit_patterns=["consider renaming"],
            learned_good_patterns=["null check before dereferencing"],
            min_confidence=0.2,
        )
    )

    scores = critic.score_comments(
        [
            {
                "node_id": "app.py::validate",
                "comment": "Consider renaming this variable for clarity.",
                "confidence": 0.5,
                "caller_count": 1,
                "validation": {"compiles": False, "tests_pass": False, "tests_attempted": False},
                "verification": {"status": "inconclusive", "ast_change_type": "unknown"},
                "verification_reason": "Fix did not compile cleanly.",
            },
            {
                "node_id": "app.py::validate",
                "comment": "Add a null check before dereferencing payload to avoid a crash.",
                "confidence": 0.5,
                "caller_count": 8,
                "reasoning": "payload can be None on empty requests",
                "validation": {"compiles": True, "tests_pass": True, "tests_attempted": True, "test_pass_ratio": 1.0},
                "verification": {"status": "passed", "ast_change_type": "logic"},
                "verification_reason": "Fix compiled, passed targeted tests (pass_ratio=1.00), and changed logic structurally.",
            },
        ],
        "bugfix in auth (behavior change expected)",
    )

    assert len(scores) == 2
    assert scores[0]["keep"] is False
    assert "compil" in scores[0]["reason"].lower()
    assert scores[1]["keep"] is True
    assert scores[1]["score"] > scores[0]["score"]


def test_critic_can_keep_high_confidence_inconclusive_comment():
    critic = Critic(strategy=ReviewStrategy(min_confidence=0.6))
    comments = [
        {
            "node_id": "auth.py::authorize",
            "comment": "When cache lookup misses, this reuses the previous principal and can authorize the wrong user.",
            "confidence": 0.92,
            "severity": "critical",
            "reasoning": "stale principal is reused on cache miss",
            "validation": {"compiles": False, "tests_pass": False, "tests_attempted": False, "compile_error_type": "TypeError"},
            "verification": {"status": "inconclusive", "ast_change_type": "unknown"},
            "verification_reason": "Fix did not compile cleanly.",
        }
    ]
    scores = [
        {
            "node_id": "auth.py::authorize",
            "comment": comments[0]["comment"],
            "score": 0.88,
            "keep": True,
            "reason": "Strongly grounded auth bug despite inconclusive fix synthesis.",
        }
    ]

    selected = critic.select_comments(comments, scores)

    assert len(selected) == 1
    assert selected[0]["critic_score"] == 0.88
    assert selected[0]["severity"] == "warning"
    assert selected[0]["original_severity"] == "critical"
    assert selected[0]["verification_survival"] == "critical_floor"


def test_critic_drops_disproved_comment_even_with_high_score():
    critic = Critic(strategy=ReviewStrategy(min_confidence=0.2))
    comments = [
        {
            "node_id": "misc.py::helper",
            "comment": "This should be renamed for clarity.",
            "confidence": 0.9,
            "validation": {"compiles": True, "tests_pass": True, "tests_attempted": True, "test_pass_ratio": 1.0},
            "verification": {"status": "disproved", "ast_change_type": "rename"},
            "verification_reason": "Fix only renamed identifiers and did not change behavior.",
        }
    ]
    scores = [
        {
            "node_id": "misc.py::helper",
            "comment": comments[0]["comment"],
            "score": 0.95,
            "keep": True,
            "reason": "High raw score.",
        }
    ]

    assert critic.select_comments(comments, scores) == []


def test_critic_prescore_penalizes_skipped_candidates():
    critic = Critic(strategy=ReviewStrategy(min_confidence=0.2))

    scores = critic.prescore_comments(
        [
            {
                "node_id": "misc.py::helper",
                "comment": "Speculative style cleanup.",
                "confidence": 0.5,
                "verification": {"status": "skipped", "ast_change_type": "unknown"},
                "validation": {},
            }
        ],
        "refactor",
    )

    assert scores[0]["score"] <= 0.25


def test_critic_backstop_keeps_best_inconclusive_comment_when_all_normal_selection_fails():
    critic = Critic(strategy=ReviewStrategy(min_confidence=0.7))
    comments = [
        {
            "node_id": "app.py::validate",
            "comment": "A missing null check can still crash this path on empty payloads.",
            "confidence": 0.61,
            "validation": {"compiles": True, "tests_attempted": False},
            "verification": {"status": "inconclusive", "ast_change_type": "logic"},
        }
    ]
    scores = [
        {
            "node_id": "app.py::validate",
            "comment": comments[0]["comment"],
            "score": 0.58,
            "keep": False,
            "reason": "Grounded but verification was inconclusive.",
        }
    ]

    selected = critic.select_comments(comments, scores)

    assert len(selected) == 1
    assert selected[0]["selection_mode"] == "backstop"
    assert selected[0]["critic_score"] == 0.58
