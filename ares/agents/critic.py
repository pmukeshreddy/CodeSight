from __future__ import annotations

import hashlib
import json
import re
from json import JSONDecodeError

from ares.agents._llm import LLMAdapter
from ares.feedback.strategy import ReviewStrategy


NIT_PATTERNS = [
    "consider renaming",
    "add docstring",
    "add a docstring",
    "missing docstring",
    "this could be more concise",
    "consider adding type hints",
    "add type hints",
    "missing type hints",
    "unused import",
    "fix formatting",
    "formatting issue",
    "formatting style",
    "inconsistent formatting",
    "trailing whitespace",
    "whitespace issue",
    "unnecessary whitespace",
    "inconsistent whitespace",
    "import ordering",
    "import order",
    "sort imports",
]

CRITICAL_SURVIVAL_CONFIDENCE = 0.85
CRITICAL_SURVIVAL_SCORE = 0.7

_RISK_KEYWORDS = {
    "null",
    "none",
    "nil",
    "deref",
    "dereference",
    "authorization",
    "authenticate",
    "sql",
    "injection",
    "overflow",
    "panic",
    "crash",
    "regression",
    "bounds",
    "out of bounds",
    "use after free",
    "race condition",
    "deadlock",
}

_STYLE_KEYWORDS = {
    "naming",
    "rename",
    "docstring",
}

_STYLE_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _STYLE_KEYWORDS) + r")\b"
)

_RISK_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _RISK_KEYWORDS) + r")\b"
)


def _has_risk_keyword(text: str) -> bool:
    """Check if any risk keyword appears in text as a whole word."""
    return bool(_RISK_KEYWORD_PATTERN.search(text))


class Critic:
    def __init__(
        self,
        api_key: str = "",
        pinecone_index=None,
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        strategy: ReviewStrategy | None = None,
        lightweight_model: str = "",
    ):
        self.client = LLMAdapter(api_key=api_key, model=model, provider=provider)
        self.lightweight_client = LLMAdapter(
            api_key=api_key,
            model=lightweight_model or model,
            provider=provider,
        )
        self.pinecone = pinecone_index
        self.strategy = strategy or ReviewStrategy()
        self.accept_threshold = max(0.6, float(self.strategy.min_confidence))
        self.high_score_threshold = min(0.9, self.accept_threshold + 0.15)
        self._cached_nit_patterns: list[str] = list(
            dict.fromkeys([*NIT_PATTERNS, *self.strategy.learned_nit_patterns])
        )
        self._normalized_nit_patterns: list[str] = [
            self._normalize_nit_text(p) for p in self._cached_nit_patterns
        ]
        self._normalized_good_patterns: list[str] = [
            self._normalize_nit_text(p) for p in self.strategy.learned_good_patterns
        ]

    def prescore_comments(self, comments: list[dict], pr_intent: str) -> list[dict]:
        if not comments:
            return []
        heuristic_scores = self._heuristic_score_comments(comments)
        normalized = self._normalize_scores(comments, heuristic_scores)
        return [self._apply_score_adjustments(comment, score) for comment, score in zip(comments, normalized)]

    # ------------------------------------------------------------------
    # Actionability classification
    # ------------------------------------------------------------------

    _ACTIONABILITY_SYSTEM_PROMPT = (
        "Classify each review comment as actionable or not_actionable.\n\n"
        "Actionable: identifies a specific, reproducible issue with the code "
        "AND points to a concrete fix or behavior change. The developer can "
        "read it and immediately know what to change.\n\n"
        "Not actionable: vague, speculative, stylistic, or hypothetical without "
        "concrete evidence. Phrases like 'consider', 'might want to', 'could "
        "potentially' without grounded evidence are not actionable.\n\n"
        "Return strict JSON:\n"
        '{"classifications":[{"id":0,"actionable":true,"reason":"..."}]}'
    )

    def classify_actionability(self, candidates: list[dict]) -> list[dict]:
        """Classify candidates as actionable/not using the lightweight LLM.

        Returns ``[{"id": int, "actionable": bool, "reason": str}, ...]``.
        Falls back to heuristic if the LLM call fails.
        """
        if not candidates:
            return []
        if not self.lightweight_client.available:
            return self._heuristic_actionability(candidates)
        entries = [
            {
                "id": i,
                "comment": c.get("comment", ""),
                "severity": c.get("severity", "warning"),
                "reasoning": c.get("reasoning", ""),
            }
            for i, c in enumerate(candidates)
        ]
        prompt = json.dumps({"candidates": entries}, separators=(",", ":"))
        max_tokens = max(300, 30 * len(candidates))
        try:
            raw = self.lightweight_client.complete(
                self._ACTIONABILITY_SYSTEM_PROMPT, prompt, max_tokens=max_tokens,
            )
            payload = json.loads(raw)
        except (JSONDecodeError, Exception):
            return self._heuristic_actionability(candidates)
        results_by_id = {
            int(c.get("id", -1)): c
            for c in payload.get("classifications", [])
        }
        output = []
        for i in range(len(candidates)):
            if i in results_by_id:
                entry = results_by_id[i]
                output.append({
                    "id": i,
                    "actionable": bool(entry.get("actionable", True)),
                    "reason": str(entry.get("reason", "")),
                })
            else:
                # Default to actionable (fail-open) if LLM missed this entry
                output.append({"id": i, "actionable": True, "reason": "Not classified."})
        return output

    _VAGUE_PATTERNS = (
        "consider ", "might want to", "could potentially",
        "you may want", "it would be nice", "perhaps",
        "think about", "looks like it could",
    )

    def _heuristic_actionability(self, candidates: list[dict]) -> list[dict]:
        """Keyword-based fallback when the lightweight LLM is unavailable."""
        output = []
        for i, c in enumerate(candidates):
            text = c.get("comment", "").lower()
            reasoning = c.get("reasoning", "")
            is_vague = any(p in text for p in self._VAGUE_PATTERNS)
            has_evidence = "Evidence:" in reasoning or "Trigger:" in reasoning
            actionable = not is_vague or has_evidence
            reason = "Vague language." if (is_vague and not has_evidence) else "Heuristic pass."
            output.append({"id": i, "actionable": actionable, "reason": reason})
        return output

    def score_comments(
        self,
        comments: list[dict],
        pr_intent: str,
        score_cache: dict[str, dict] | None = None,
    ) -> list[dict]:
        if not comments:
            return []
        # Split into cached hits and uncached misses to avoid redundant LLM calls.
        if score_cache is not None:
            cached_results: list[tuple[int, dict]] = []
            uncached: list[tuple[int, dict]] = []
            for i, comment in enumerate(comments):
                key = self._score_cache_key(comment, pr_intent)
                if key in score_cache:
                    cached_results.append((i, score_cache[key]))
                else:
                    uncached.append((i, comment))
            if not uncached:
                return [cached_results[i][1] for i in range(len(cached_results))]
            uncached_comments = [c for _, c in uncached]
            llm_scores = self._llm_score_comments(uncached_comments, pr_intent)
            if not llm_scores:
                llm_scores = self._heuristic_score_comments(uncached_comments)
            normalized = self._normalize_scores(uncached_comments, llm_scores)
            uncached_adjusted = [
                self._apply_score_adjustments(comment, score)
                for comment, score in zip(uncached_comments, normalized)
            ]
            # Store new results in cache.
            for (_, comment), adjusted in zip(uncached, uncached_adjusted):
                key = self._score_cache_key(comment, pr_intent)
                score_cache[key] = adjusted
            # Merge cached and new results in original order.
            final: list[dict] = [{}] * len(comments)
            for i, result in cached_results:
                final[i] = result
            for (i, _), result in zip(uncached, uncached_adjusted):
                final[i] = result
            return final
        llm_scores = self._llm_score_comments(comments, pr_intent)
        if not llm_scores:
            llm_scores = self._heuristic_score_comments(comments)
        normalized = self._normalize_scores(comments, llm_scores)
        return [self._apply_score_adjustments(comment, score) for comment, score in zip(comments, normalized)]

    def _score_cache_key(self, comment: dict, pr_intent: str) -> str:
        """Build a stable cache key from the fields that influence LLM scoring."""
        verification = comment.get("verification", {})
        validation = comment.get("validation", {})
        raw = "|".join([
            comment.get("comment", ""),
            comment.get("severity", ""),
            str(comment.get("confidence", 0.0)),
            comment.get("reasoning", ""),
            verification.get("status", ""),
            verification.get("mode", ""),
            str(validation.get("compiles", "")),
            str(validation.get("tests_pass", "")),
            str(validation.get("test_pass_ratio", "")),
            validation.get("repro", {}).get("status", ""),
            pr_intent,
        ])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def select_comments(
        self,
        comments: list[dict],
        scores: list[dict],
        min_score: float | None = None,
    ) -> list[dict]:
        threshold = self.accept_threshold if min_score is None else min_score
        selected = []
        for comment, score in zip(comments, scores):
            status = comment.get("verification", {}).get("status", "unknown")
            if status == "disproved":
                continue
            if self._survives_critical_floor(comment, score, status):
                selected.append(
                    {
                        **comment,
                        "severity": "warning",
                        "original_severity": comment.get("severity", "warning"),
                        "verification_survival": "critical_floor",
                        "critic_score": float(score.get("score", 0.0)),
                        "critique_reason": score.get("reason", ""),
                        "confidence": min(
                            1.0,
                            max(float(comment.get("confidence", 0.0)), float(score.get("score", 0.0))),
                        ),
                    }
                )
                continue
            effective_threshold = threshold if status == "passed" else min(0.85, threshold + 0.1)
            if float(score.get("score", 0.0)) < effective_threshold:
                continue
            selected.append(
                {
                    **comment,
                    "critic_score": float(score.get("score", 0.0)),
                    "critique_reason": score.get("reason", ""),
                    "confidence": min(
                        1.0,
                        (float(comment.get("confidence", 0.0)) + float(score.get("score", 0.0))) / 2,
                    ),
                }
            )
        if selected:
            return selected
        return self._fallback_select_comments(comments, scores, threshold)

    def all_scores_high(self, scores: list[dict]) -> bool:
        return bool(scores) and all(float(score.get("score", 0.0)) >= self.high_score_threshold for score in scores)

    def average_score(self, scores: list[dict]) -> float:
        if not scores:
            return 0.0
        return sum(float(score.get("score", 0.0)) for score in scores) / len(scores)

    _SCORE_SYSTEM_PROMPT = """
You are critiquing draft code review comments. Score each comment from 0.0 to 1.0.
High scores (0.7-1.0): concrete bugs, logic errors, unhandled edge cases, security issues, correctness problems, or behavioral regressions grounded in the diff.
Medium scores (0.4-0.7): plausible concerns with partial evidence or missing error handling for likely scenarios.
Low scores (0.0-0.4): naming, formatting, style, docstrings, speculative advice, or weak evidence.
Return strict JSON:
{"scores":[{"id":0,"score":0.0,"reason":"...","keep":false}]}
""".strip()

    def build_score_request(self, comments: list[dict], pr_intent: str) -> tuple[str, str, int] | None:
        """Build ``(system_prompt, user_prompt, max_tokens)`` for batch API scoring."""
        if not comments:
            return None
        prompt = self._build_score_prompt(comments, pr_intent)
        max_tokens = min(4096, max(500, 40 * len(comments)))
        return (self._SCORE_SYSTEM_PROMPT, prompt, max_tokens)

    def parse_score_response(self, comments: list[dict], raw: str) -> list[dict]:
        """Parse a raw LLM scoring response into score dicts."""
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except JSONDecodeError:
            return []
        llm_scores = payload.get("scores", [])
        if not llm_scores:
            llm_scores = self._heuristic_score_comments(comments)
        normalized = self._normalize_scores(comments, llm_scores)
        return [self._apply_score_adjustments(comment, score) for comment, score in zip(comments, normalized)]

    def _build_score_prompt(self, comments: list[dict], pr_intent: str) -> str:
        comment_entries = []
        for index, comment in enumerate(comments):
            verification = comment.get("verification", {})
            validation = comment.get("validation", {})
            entry: dict = {
                "id": index,
                "comment": comment.get("comment", ""),
                "severity": comment.get("severity", "warning"),
                "confidence": comment.get("confidence", 0.0),
            }
            if comment.get("reasoning"):
                entry["reasoning"] = comment["reasoning"]
            if comment.get("risk", "standard") != "standard":
                entry["risk"] = comment["risk"]
            if comment.get("caller_count", 0):
                entry["caller_count"] = comment["caller_count"]
            v: dict = {}
            v_status = verification.get("status", "unknown")
            if v_status != "unknown":
                v["status"] = v_status
            v_mode = verification.get("mode", "unknown")
            if v_mode != "unknown":
                v["mode"] = v_mode
            if comment.get("verification_reason"):
                v["reason"] = comment["verification_reason"]
            ast_type = verification.get("ast_change_type", "unknown")
            if ast_type != "unknown":
                v["ast_change_type"] = ast_type
            if validation.get("compiles"):
                v["compiled"] = True
            if validation.get("compile_error"):
                v["compile_error"] = validation["compile_error"]
            if validation.get("tests_pass"):
                v["tests_passed"] = True
            if validation.get("test_pass_ratio") is not None:
                v["test_pass_ratio"] = validation["test_pass_ratio"]
            if validation.get("test_error"):
                v["test_error"] = validation["test_error"]
            repro_status = validation.get("repro", {}).get("status", "")
            if repro_status:
                v["repro_status"] = repro_status
            if v:
                entry["verification"] = v
            comment_entries.append(entry)
        payload: dict = {"pr_intent": pr_intent, "comments": comment_entries}
        guidance: dict = {}
        if self.strategy.extra_reviewer_instructions:
            guidance["extra_instructions"] = self.strategy.extra_reviewer_instructions
        if self.strategy.learned_nit_patterns:
            guidance["learned_nit_patterns"] = self.strategy.learned_nit_patterns
        if self.strategy.learned_good_patterns:
            guidance["learned_good_patterns"] = self.strategy.learned_good_patterns
        if guidance:
            payload["strategy_guidance"] = guidance
        return json.dumps(payload, separators=(",", ":"))

    def _llm_score_comments(self, comments: list[dict], pr_intent: str) -> list[dict]:
        if not self.lightweight_client.available:
            return []
        prompt = self._build_score_prompt(comments, pr_intent)
        max_tokens = min(4096, max(500, 40 * len(comments)))
        raw = self.lightweight_client.complete(self._SCORE_SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
        try:
            payload = json.loads(raw)
        except JSONDecodeError:
            return []
        return payload.get("scores", [])

    def _apply_verification_adjustments(self, comment: dict, score: float) -> tuple[float, list[str]]:
        """Shared logic: adjust *score* based on verification/validation status.

        Returns ``(adjusted_score, [reason_strings])``.
        """
        reasons: list[str] = []
        verification = comment.get("verification", {})
        validation = comment.get("validation", {})
        status = verification.get("status", "unknown")
        test_ratio = self._test_pass_ratio(comment)
        compile_error_type = validation.get("compile_error_type") or validation.get("compile", {}).get("error_type", "")
        test_error_type = validation.get("test_error_type") or validation.get("tests", {}).get("error_type", "")
        repro_status = validation.get("repro", {}).get("status", "")
        if status == "passed":
            if repro_status == "confirmed_fixed":
                score = max(score, 0.85)
                reasons.append("Verification confirmed the bug with a generated repro test.")
            elif validation.get("tests_attempted"):
                score = max(score, 0.75)
                reasons.append("Verification succeeded with a compiling fix that passed targeted tests.")
            else:
                score = max(score, 0.65)
                reasons.append("Verification succeeded with a compiling structural fix, but no targeted tests ran.")
        elif status == "inconclusive":
            if compile_error_type in {"SyntaxError", "IndentationError", "ParseError"}:
                score = min(score, 0.55)
                reasons.append("Verification only failed with a syntax error in the synthesized fix.")
            elif compile_error_type == "TimeoutError":
                score = min(score, 0.1)
                reasons.append("Verification timed out.")
            elif compile_error_type == "TypeError":
                score = min(score, 0.25)
                reasons.append("Verification introduced a type error.")
            elif validation.get("compiles") is False:
                score = min(score, 0.35)
                reasons.append("Verification could not produce a compiling fix.")
            elif validation.get("tests_attempted") and validation.get("tests_pass") is False:
                cap = 0.4 if test_ratio is None else max(0.3, min(0.55, 0.25 + (0.4 * test_ratio)))
                score = min(score, cap)
                if test_error_type == "AssertionError":
                    score = min(1.0, score + 0.1)
                    reasons.append("Verification failed by assertion, which still supports the bug claim.")
                else:
                    reasons.append("Verification failed targeted tests.")
            else:
                score = min(score, 0.6)
                reasons.append("Verification was inconclusive.")
            if repro_status == "confirmed_unfixed":
                score = min(1.0, score + 0.05)
                reasons.append("Generated repro still fails after the synthesized fix.")
            elif repro_status == "not_reproduced":
                score = min(score, 0.3)
                reasons.append("Generated repro did not reproduce the reported bug.")
        elif status == "skipped":
            score = min(score, 0.25)
            reasons.append("Candidate was skipped by verification routing.")
        elif status == "disproved":
            score = min(score, 0.1)
            reasons.append("Verification only found cosmetic or rename-only changes.")
        ast_change_type = verification.get("ast_change_type", "unknown")
        if ast_change_type in {"cosmetic", "rename"}:
            score = min(score, 0.1)
            reasons.append("Verification showed a cosmetic or rename-only fix.")
        if isinstance(test_ratio, float) and test_ratio >= 0.8:
            score = min(1.0, score + 0.05)
            reasons.append("Most targeted tests passed under the synthesized fix.")
        return score, reasons

    def _heuristic_score_comments(self, comments: list[dict]) -> list[dict]:
        scores = []
        for index, comment in enumerate(comments):
            text = comment.get("comment", "")
            normalized = self._normalize_nit_text(text)
            score = max(0.1, float(comment.get("confidence", 0.0)))
            reasons = []
            is_nit = self._is_seeded_nit(text)
            if is_nit:
                score -= 0.45
                reasons.append("Matches a learned nit/style pattern.")
            elif _STYLE_KEYWORD_PATTERN.search(normalized):
                score -= 0.2
                reasons.append("Reads as style or naming feedback.")
            if self._matches_learned_good_pattern(text):
                score += 0.2
                reasons.append("Matches a learned high-value pattern.")
            if _has_risk_keyword(normalized):
                score += 0.15
                reasons.append("Points to a concrete runtime or security risk.")
            reasoning = comment.get("reasoning", "")
            if reasoning:
                score += 0.05
                reasons.append("Includes explicit reasoning.")
                # Bonus for structured reasoning with concrete evidence
                _STRUCTURED_FIELDS = ("Premise:", "Evidence:", "Trigger:", "Impact:")
                fields_present = sum(1 for f in _STRUCTURED_FIELDS if f in reasoning)
                if fields_present >= 3:
                    score += 0.05
                    reasons.append("Well-structured reasoning with concrete evidence.")
            if float(comment.get("caller_count", 0)) >= 5:
                score += 0.05
                reasons.append("Touches code with broader blast radius.")
            score = max(0.0, min(1.0, score))
            scores.append(
                {
                    "id": index,
                    "score": round(score, 3),
                    "reason": " ".join(reasons) or "No strong signals found.",
                    "keep": score >= self.accept_threshold,
                }
            )
        return scores

    def _normalize_scores(self, comments: list[dict], scores: list[dict]) -> list[dict]:
        by_id = {
            int(score.get("id", -1)): score
            for score in scores
            if isinstance(score, dict)
        }
        # Batch-score all missing comments in one call instead of per-comment.
        missing = [(i, comments[i]) for i in range(len(comments)) if i not in by_id]
        fallbacks: dict[int, dict] = {}
        if missing:
            missing_comments = [c for _, c in missing]
            heuristic = self._heuristic_score_comments(missing_comments)
            for (idx, _), fb in zip(missing, heuristic):
                fb["id"] = idx
                fallbacks[idx] = fb
        normalized = []
        for index in range(len(comments)):
            if index in fallbacks:
                normalized.append(fallbacks[index])
            else:
                score = by_id[index]
                normalized.append(
                    {
                        "id": index,
                        "score": max(0.0, min(1.0, float(score.get("score", 0.0)))),
                        "reason": str(score.get("reason", "")).strip(),
                        "keep": bool(score.get("keep", False)),
                    }
                )
        return normalized

    def _apply_score_adjustments(self, comment: dict, score: dict) -> dict:
        text = comment.get("comment", "")
        value = float(score.get("score", 0.0))
        reasons = [score.get("reason", "").strip()] if score.get("reason", "").strip() else []
        value, verification_reasons = self._apply_verification_adjustments(comment, value)
        reasons.extend(verification_reasons)
        if self._is_seeded_nit(text):
            value = min(value, self.accept_threshold - 0.1)
            reasons.append("Downweighted by nit-pattern match.")
        if self._matches_learned_good_pattern(text):
            value = min(1.0, value + 0.1)
            reasons.append("Boosted by learned good pattern.")
        if self.pinecone is not None:
            matches = self.pinecone.query_similar(text, top_k=5)
            downvotes = [m for m in matches if m.get("score", 0.0) > 0.85 and m.get("label") == "downvote"]
            upvotes = [m for m in matches if m.get("score", 0.0) > 0.85 and m.get("label") == "upvote"]
            if len(downvotes) >= 3:
                # Hard kill — same comment ignored 3+ times historically
                value = 0.0
                reasons.append("Similar comments were historically ignored (3+ downvotes).")
            elif downvotes:
                value = max(0.0, value - min(0.25, 0.08 * len(downvotes)))
                reasons.append("Similar comments were historically ignored.")
            if upvotes:
                value = min(1.0, value + min(0.2, 0.06 * len(upvotes)))
                reasons.append("Similar comments were historically addressed.")
        value = round(max(0.0, min(1.0, value)), 3)
        deduped_reasons = list(dict.fromkeys(reason for reason in reasons if reason))
        return {
            "id": score.get("id", 0),
            "node_id": comment.get("node_id", ""),
            "comment": text,
            "score": value,
            "keep": value >= self.accept_threshold,
            "reason": " ".join(deduped_reasons).strip() or "No critique reason available.",
        }

    def _is_seeded_nit(self, text: str) -> bool:
        return self._contains_nit_pattern(text)

    def _contains_nit_pattern(self, text: str) -> bool:
        normalized_text = self._normalize_nit_text(text)
        return any(p in normalized_text for p in self._normalized_nit_patterns)

    def _matches_learned_good_pattern(self, text: str) -> bool:
        normalized_text = self._normalize_nit_text(text)
        return any(p in normalized_text for p in self._normalized_good_patterns)

    def _normalize_nit_text(self, text: str) -> str:
        lowered = text.lower().replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", lowered).strip()

    def _test_pass_ratio(self, comment: dict) -> float | None:
        validation = comment.get("validation", {})
        ratio = validation.get("test_pass_ratio")
        if isinstance(ratio, (int, float)):
            return float(ratio)
        nested_ratio = validation.get("tests", {}).get("pass_ratio")
        if isinstance(nested_ratio, (int, float)):
            return float(nested_ratio)
        return None

    def _survives_critical_floor(self, comment: dict, score: dict, status: str) -> bool:
        if status in {"passed", "disproved"}:
            return False
        if str(comment.get("severity", "warning")).lower() != "critical":
            return False
        if float(comment.get("confidence", 0.0)) < CRITICAL_SURVIVAL_CONFIDENCE:
            return False
        return float(score.get("score", 0.0)) >= max(self.accept_threshold, CRITICAL_SURVIVAL_SCORE)

    def _nit_patterns(self) -> list[str]:
        return self._cached_nit_patterns

    def _fallback_select_comments(
        self,
        comments: list[dict],
        scores: list[dict],
        threshold: float,
    ) -> list[dict]:
        candidates: list[tuple[float, float, dict, dict]] = []
        floor = max(0.45, threshold - 0.15)
        for comment, score in zip(comments, scores):
            verification = comment.get("verification", {})
            status = verification.get("status", "unknown")
            if status not in {"passed", "inconclusive"}:
                continue
            if verification.get("ast_change_type") in {"cosmetic", "rename"}:
                continue
            score_value = float(score.get("score", 0.0))
            if score_value < floor:
                continue
            confidence = float(comment.get("confidence", 0.0))
            candidates.append((score_value, confidence, comment, score))
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                1 if item[2].get("verification", {}).get("status") == "passed" else 0,
            ),
            reverse=True,
        )
        selected = []
        for score_value, _, comment, score in candidates[:2]:
            selected.append(
                {
                    **comment,
                    "critic_score": score_value,
                    "critique_reason": score.get("reason", ""),
                    "confidence": min(
                        1.0,
                        max(float(comment.get("confidence", 0.0)), score_value),
                    ),
                    "selection_mode": "backstop",
                }
            )
        return selected
