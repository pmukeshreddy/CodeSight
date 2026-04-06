"""Tests for the heuristic scoring fixes in Critic._heuristic_score_comments.

Covers:
  1. NIT_PATTERNS no longer match bare 'formatting'/'whitespace' inside real comments
  2. Risk keywords use word boundaries (no substring false positives)
  3. No double penalty (nit pattern + style keyword on the same comment)
  4. Removed false-positive-prone boost keywords ('token', 'empty')
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ares.agents.critic import Critic, _has_risk_keyword


def _make_comment(text: str, confidence: float = 0.5, **extras) -> dict:
    return {"comment": text, "confidence": confidence, "reasoning": "", **extras}


def _score(critic: Critic, text: str, confidence: float = 0.5, **extras) -> dict:
    comments = [_make_comment(text, confidence, **extras)]
    return critic._heuristic_score_comments(comments)[0]


def _build_critic() -> Critic:
    return Critic(api_key="", strategy=None)


# ── Fix 1: NIT_PATTERNS specificity ─────────────────────────────────────

class TestNitPatternSpecificity:
    """Bare words 'formatting' and 'whitespace' should NOT trigger nit penalty
    when they appear in legitimate security/bug comments."""

    def test_whitespace_in_security_comment_not_penalized(self):
        critic = _build_critic()
        result = _score(critic, "This function doesn't sanitize user input, allowing whitespace injection in SQL queries")
        # Should NOT get the -0.45 nit penalty
        assert not critic._is_seeded_nit("This function doesn't sanitize user input, allowing whitespace injection in SQL queries")
        assert result["score"] > 0.3, f"Security comment about whitespace injection scored too low: {result['score']}"

    def test_formatting_in_security_comment_not_penalized(self):
        critic = _build_critic()
        result = _score(critic, "The formatting string %s is vulnerable to format string attacks")
        assert not critic._is_seeded_nit("The formatting string %s is vulnerable to format string attacks")
        assert result["score"] > 0.3, f"Security comment about format strings scored too low: {result['score']}"

    def test_actual_formatting_nit_still_penalized(self):
        critic = _build_critic()
        assert critic._is_seeded_nit("Fix formatting issue in this function")

    def test_actual_whitespace_nit_still_penalized(self):
        critic = _build_critic()
        assert critic._is_seeded_nit("Remove trailing whitespace")

    def test_inconsistent_formatting_nit_still_penalized(self):
        critic = _build_critic()
        assert critic._is_seeded_nit("Inconsistent formatting in this block")

    def test_whitespace_issue_nit_still_penalized(self):
        critic = _build_critic()
        assert critic._is_seeded_nit("There is a whitespace issue here")

    def test_bare_whitespace_word_not_a_nit(self):
        """A comment mentioning 'whitespace' in a non-nit context should not be penalized."""
        critic = _build_critic()
        assert not critic._is_seeded_nit("Whitespace characters in the input can cause parsing failures")

    def test_bare_formatting_word_not_a_nit(self):
        critic = _build_critic()
        assert not critic._is_seeded_nit("The formatting of the error message is incorrect and may confuse users")


# ── Fix 2: Risk keyword false positives removed ─────────────────────────

class TestRiskKeywordFalsePositives:
    """'token' and 'empty' were removed from risk keywords because they
    appear too often in non-security contexts."""

    def test_token_does_not_boost(self):
        critic = _build_critic()
        result = _score(critic, "Consider extracting this token parsing into a separate method")
        assert "runtime or security risk" not in result.get("reason", ""), \
            "'token' in a style comment should not trigger security boost"

    def test_empty_does_not_boost(self):
        critic = _build_critic()
        result = _score(critic, "This could return an empty list instead of a populated one")
        assert "runtime or security risk" not in result.get("reason", ""), \
            "'empty' in a style comment should not trigger security boost"

    def test_null_still_boosts(self):
        critic = _build_critic()
        result = _score(critic, "Potential null dereference when user is not authenticated")
        assert "runtime or security risk" in result.get("reason", ""), \
            "'null' should still trigger security boost"

    def test_sql_injection_still_boosts(self):
        critic = _build_critic()
        result = _score(critic, "This query is vulnerable to SQL injection")
        assert "runtime or security risk" in result.get("reason", ""), \
            "'sql' and 'injection' should trigger security boost"

    def test_overflow_still_boosts(self):
        critic = _build_critic()
        result = _score(critic, "Integer overflow possible with large input values")
        assert "runtime or security risk" in result.get("reason", ""), \
            "'overflow' should trigger security boost"

    def test_race_condition_boosts(self):
        critic = _build_critic()
        result = _score(critic, "This has a race condition between the check and the write")
        assert "runtime or security risk" in result.get("reason", ""), \
            "'race condition' should trigger security boost"

    def test_deadlock_boosts(self):
        critic = _build_critic()
        result = _score(critic, "Acquiring locks in this order can cause a deadlock")
        assert "runtime or security risk" in result.get("reason", ""), \
            "'deadlock' should trigger security boost"


# ── Fix 3: No double penalty ────────────────────────────────────────────

class TestNoDoublePenalty:
    """When a comment matches a NIT_PATTERN, the style keyword penalty
    should NOT also apply. Only one penalty path fires."""

    def test_consider_renaming_single_penalty(self):
        critic = _build_critic()
        result = _score(critic, "Consider renaming this variable to something more descriptive", confidence=0.5)
        # NIT_PATTERN match: -0.45 from 0.5 = 0.05, clamped to max(0.1, ...)
        # Should NOT also subtract -0.2 for 'rename' keyword
        # With double penalty: 0.5 - 0.45 - 0.2 = -0.15 -> clamped 0.0
        # Without double penalty: 0.5 - 0.45 = 0.05 -> clamped 0.1
        assert result["score"] >= 0.05, \
            f"Double penalty applied: score={result['score']}, expected >= 0.05"

    def test_add_docstring_no_double_penalty(self):
        critic = _build_critic()
        result = _score(critic, "Add docstring to explain the parameters", confidence=0.5)
        # NIT_PATTERN "add docstring" matches, so 'docstring' style keyword should not also fire
        reasons = result.get("reason", "")
        nit_count = reasons.count("nit/style pattern")
        style_count = reasons.count("style or naming feedback")
        assert not (nit_count > 0 and style_count > 0), \
            f"Both nit and style penalties applied: {reasons}"

    def test_style_keyword_without_nit_pattern_still_penalizes(self):
        """If the comment has a style keyword but does NOT match any NIT_PATTERN,
        the style penalty should still apply."""
        critic = _build_critic()
        result = _score(critic, "You should rename the database handle for clarity", confidence=0.7)
        # "rename" is a style keyword, but "you should rename the database handle..."
        # does NOT match any NIT_PATTERN (no "consider renaming" substring)
        assert "style or naming feedback" in result.get("reason", ""), \
            "Style keyword penalty should fire when no nit pattern matches"


# ── Fix 4: Word boundary matching ───────────────────────────────────────

class TestWordBoundaryMatching:
    """Risk keywords should only match as whole words, not as substrings."""

    def test_none_does_not_match_inside_nonetheless(self):
        assert not _has_risk_keyword("this is nonetheless important")

    def test_none_does_not_match_inside_component(self):
        assert not _has_risk_keyword("the component handles rendering")

    def test_nil_does_not_match_inside_vanilla(self):
        assert not _has_risk_keyword("use vanilla javascript here")

    def test_null_matches_as_whole_word(self):
        assert _has_risk_keyword("potential null pointer dereference")

    def test_none_matches_as_whole_word(self):
        assert _has_risk_keyword("returns none when it should return a value")

    def test_sql_matches_as_whole_word(self):
        assert _has_risk_keyword("vulnerable to sql injection")

    def test_crash_matches_as_whole_word(self):
        assert _has_risk_keyword("this will crash on invalid input")

    def test_bounds_does_not_match_inside_rebounds(self):
        assert not _has_risk_keyword("the value rebounds quickly")

    def test_bounds_matches_out_of_bounds(self):
        assert _has_risk_keyword("array index out of bounds access")

    def test_panic_does_not_match_inside_hispanic(self):
        assert not _has_risk_keyword("hispanic users affected")

    def test_panic_matches_as_whole_word(self):
        assert _has_risk_keyword("this will panic at runtime")


# ── Integration: full scoring pipeline ──────────────────────────────────

class TestFullScoringPipeline:
    """End-to-end tests that verify the combined effect of all fixes."""

    def test_security_comment_with_whitespace_word_scores_well(self):
        """A real security finding mentioning 'whitespace' should score high,
        not get killed by the old bare 'whitespace' nit pattern."""
        critic = _build_critic()
        result = _score(
            critic,
            "User input containing whitespace is not sanitized before SQL query injection",
            confidence=0.8,
            reasoning="The function passes raw input to the query builder.",
        )
        # Should get: +0.15 (risk keywords: injection, sql) + 0.05 (reasoning)
        # Should NOT get: -0.45 (nit) or -0.2 (style)
        assert result["score"] >= 0.7, \
            f"Legitimate security finding scored too low: {result['score']}"

    def test_pure_style_comment_scores_low(self):
        critic = _build_critic()
        result = _score(
            critic,
            "Consider renaming this variable to a more descriptive name",
            confidence=0.3,
        )
        assert result["score"] < 0.3, \
            f"Pure style comment scored too high: {result['score']}"

    def test_bug_comment_without_old_token_keyword(self):
        """'token' was removed from risk keywords. A real bug about tokens
        should still score well based on other signals (confidence, reasoning)."""
        critic = _build_critic()
        result = _score(
            critic,
            "The authentication token is never validated against the revocation list, "
            "allowing expired tokens to be reused. This is a regression from the previous behavior.",
            confidence=0.85,
            reasoning="Token validation was removed in this diff.",
        )
        # 'regression' is still a risk keyword, so +0.15 still fires
        # confidence 0.85 + 0.15 (risk) + 0.05 (reasoning) = 1.0 (capped)
        assert result["score"] >= 0.8, \
            f"Real bug comment scored too low even without 'token' boost: {result['score']}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
