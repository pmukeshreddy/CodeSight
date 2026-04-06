from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "warning": 0.7,
    "suggestion": 0.4,
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
}


@dataclass(slots=True)
class ReviewStrategy:
    bug_fix_freq_critical: int = 3
    change_freq_utility: int = 1
    utility_blast_radius_threshold: int = 5
    extra_reviewer_instructions: list[str] = field(default_factory=list)
    learned_nit_patterns: list[str] = field(default_factory=list)
    learned_good_patterns: list[str] = field(default_factory=list)
    severity_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SEVERITY_WEIGHTS)
    )
    min_confidence: float = 0.15
    version: int = 1

    @classmethod
    def load(cls, repo_path: str) -> "ReviewStrategy":
        path = cls.strategy_path(repo_path)
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict) -> "ReviewStrategy":
        severity_weights = dict(DEFAULT_SEVERITY_WEIGHTS)
        severity_weights.update(
            {
                str(key).lower(): float(value)
                for key, value in (payload.get("severity_weights") or {}).items()
            }
        )
        return cls(
            bug_fix_freq_critical=int(payload.get("bug_fix_freq_critical", 3)),
            change_freq_utility=int(payload.get("change_freq_utility", 1)),
            utility_blast_radius_threshold=int(payload.get("utility_blast_radius_threshold", 5)),
            extra_reviewer_instructions=[
                str(item).strip()
                for item in payload.get("extra_reviewer_instructions", [])
                if str(item).strip()
            ],
            learned_nit_patterns=[
                str(item).strip()
                for item in payload.get("learned_nit_patterns", [])
                if str(item).strip()
            ],
            learned_good_patterns=[
                str(item).strip()
                for item in payload.get("learned_good_patterns", [])
                if str(item).strip()
            ],
            severity_weights=severity_weights,
            min_confidence=float(payload.get("min_confidence", 0.15)),
            version=int(payload.get("version", 1)),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["severity_weights"] = {
            str(key).lower(): float(value)
            for key, value in self.severity_weights.items()
        }
        return payload

    def save(self, repo_path: str) -> str:
        path = self.strategy_path(repo_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(path)

    @classmethod
    def strategy_path(cls, repo_path: str) -> Path:
        return cls.storage_dir(repo_path) / "strategy.json"

    @staticmethod
    def storage_dir(repo_path: str) -> Path:
        base = Path(os.path.abspath(repo_path or os.getcwd()))
        return base / ".ares"
