from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from ares.agents._llm import LLMAdapter
from ares.feedback.strategy import ReviewStrategy
from ares.utils.json_utils import parse_llm_json


class FeedbackLearner:
    def __init__(
        self,
        repo_path: str,
        strategy: ReviewStrategy | None = None,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
    ):
        self.repo_path = os.path.abspath(repo_path or os.getcwd())
        self.strategy = strategy or ReviewStrategy.load(self.repo_path)
        self.client = LLMAdapter(api_key=api_key, model=model, provider=provider)

    def set_repo_path(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path or os.getcwd())
        self.strategy = ReviewStrategy.load(self.repo_path)

    def improve(self) -> dict:
        outcomes = self._load_outcomes()
        if not outcomes:
            return {
                "updated": False,
                "reason": "no_outcomes",
                "strategy": self.strategy.to_dict(),
            }
        stats = self._aggregate(outcomes)
        updated_strategy = self._generate_strategy(stats)
        updated_strategy.save(self.repo_path)
        self.strategy = updated_strategy
        return {
            "updated": True,
            "outcomes_analyzed": len(outcomes),
            "strategy": updated_strategy.to_dict(),
            "stats": stats,
        }

    def _generate_strategy(self, stats: dict) -> ReviewStrategy:
        if not self.client.available:
            return self.strategy
        system_prompt = """
You improve a code-review agent strategy using aggregate feedback outcomes.
Return strict JSON with any subset of:
{
  "bug_fix_freq_critical": 3,
  "change_freq_utility": 1,
  "utility_blast_radius_threshold": 5,
  "extra_reviewer_instructions": ["..."],
  "learned_nit_patterns": ["..."],
  "learned_good_patterns": ["..."],
  "severity_weights": {"critical": 1.0, "warning": 0.7, "suggestion": 0.4},
  "min_confidence": 0.2
}

Prefer small, evidence-based changes. Do not invent large prompt rules unless the stats justify them.
""".strip()
        prompt = json.dumps(
            {
                "current_strategy": self.strategy.to_dict(),
                "feedback_stats": stats,
            },
            indent=2,
            sort_keys=True,
        )
        raw = self.client.complete(system_prompt, prompt, max_tokens=1600)
        payload = self._parse_json(raw)
        if not payload:
            return self.strategy
        next_strategy = ReviewStrategy.from_dict(
            {
                **self.strategy.to_dict(),
                **payload,
                "version": max(
                    int(self.strategy.version) + 1,
                    int(payload.get("version", int(self.strategy.version) + 1)),
                ),
            }
        )
        return next_strategy

    def _aggregate(self, outcomes: list[dict]) -> dict:
        summary = {
            "total": len(outcomes),
            "addressed": sum(1 for item in outcomes if item.get("status") == "addressed"),
            "ignored": sum(1 for item in outcomes if item.get("status") == "ignored"),
        }
        grouped = {
            "severity": self._group_rates(outcomes, "severity"),
            "intent": self._group_rates(outcomes, "pr_intent"),
            "risk": self._group_rates(outcomes, "risk"),
            "source": self._group_rates(outcomes, "source"),
        }
        examples = {
            "addressed_patterns": self._pattern_examples(outcomes, "addressed"),
            "ignored_patterns": self._pattern_examples(outcomes, "ignored"),
        }
        return {
            "summary": summary,
            "grouped_rates": grouped,
            "examples": examples,
        }

    def _group_rates(self, outcomes: list[dict], key: str) -> dict[str, dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in outcomes:
            grouped[str(item.get(key) or "unknown")].append(item)
        result = {}
        for group_key, items in grouped.items():
            total = len(items)
            addressed = sum(1 for item in items if item.get("status") == "addressed")
            ignored = sum(1 for item in items if item.get("status") == "ignored")
            result[group_key] = {
                "total": total,
                "address_rate": round(addressed / total, 3) if total else 0.0,
                "ignored_rate": round(ignored / total, 3) if total else 0.0,
            }
        return result

    def _pattern_examples(self, outcomes: list[dict], status: str) -> list[dict]:
        counter: Counter[str] = Counter()
        samples: dict[str, str] = {}
        for item in outcomes:
            if item.get("status") != status:
                continue
            signature = self._comment_signature(item.get("comment", ""))
            if not signature:
                continue
            counter[signature] += 1
            samples.setdefault(signature, item.get("comment", ""))
        return [
            {"pattern": signature, "count": count, "example": samples.get(signature, "")}
            for signature, count in counter.most_common(5)
        ]

    def _comment_signature(self, comment: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", (comment or "").lower())
        return " ".join(tokens[:8])

    def _load_outcomes(self) -> list[dict]:
        path = ReviewStrategy.storage_dir(self.repo_path) / "feedback_outcomes.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _parse_json(self, raw: str) -> dict:
        return parse_llm_json(raw)
