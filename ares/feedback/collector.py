from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ares.feedback.strategy import ReviewStrategy


class FeedbackCollector:
    def __init__(self, repo_path: str, github_client=None, pinecone_client=None):
        self.repo_path = os.path.abspath(repo_path or os.getcwd())
        self.github = github_client
        self.pinecone = pinecone_client

    def set_repo_path(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path or os.getcwd())

    def record_posted_comments(
        self,
        repo_name: str,
        pr_number: int,
        comments: list[dict],
    ) -> list[dict]:
        if not comments:
            return []
        pending = self._load_records(self.pending_path)
        existing_ids = {item.get("id") for item in pending}
        posted_at = datetime.now(timezone.utc).isoformat()
        new_records: list[dict] = []
        for index, comment in enumerate(comments):
            comment_text = (comment.get("comment") or comment.get("message") or "").strip()
            if not comment_text:
                continue
            record = {
                "id": self._feedback_id(repo_name, pr_number, comment, posted_at, index),
                "repo_name": repo_name,
                "pr_number": pr_number,
                "file": comment.get("file", ""),
                "line_start": int(comment.get("line_start", comment.get("line", 0)) or 0),
                "line_end": int(comment.get("line_end", comment.get("line", 0)) or 0),
                "comment": comment_text,
                "severity": comment.get("severity", "warning"),
                "source": comment.get("source", "llm"),
                "risk": comment.get("risk", comment.get("original_risk", "unknown")),
                "pr_intent": comment.get("pr_intent", ""),
                "posted_at": posted_at,
                "status": "pending",
            }
            if record["id"] in existing_ids:
                continue
            new_records.append(record)
            existing_ids.add(record["id"])
        if new_records:
            pending.extend(new_records)
            self._save_records(self.pending_path, pending)
        return new_records

    def collect_feedback(self) -> dict:
        pending = self._load_records(self.pending_path)
        if not pending:
            return {"pending": 0, "resolved": 0, "outcomes": []}
        if self.github is None:
            return {"pending": len(pending), "resolved": 0, "outcomes": []}
        outcomes = self._load_records(self.outcomes_path)
        outcome_ids = {item.get("id") for item in outcomes}
        remaining: list[dict] = []
        resolved: list[dict] = []
        grouped: dict[tuple[str, int], list[dict]] = {}
        for record in pending:
            grouped.setdefault((record["repo_name"], int(record["pr_number"])), []).append(record)
        for (repo_name, pr_number), records in grouped.items():
            statuses = self.github.get_pending_comment_feedback(repo_name, pr_number, records)
            for item in statuses:
                if item.get("outcome") == "pending":
                    remaining.append(item)
                    continue
                finalized = {
                    **item,
                    "status": item.get("outcome", "pending"),
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                }
                resolved.append(finalized)
                if finalized["id"] not in outcome_ids:
                    outcomes.append(finalized)
                    outcome_ids.add(finalized["id"])
                    if self.pinecone is not None:
                        self.pinecone.upsert_feedback(
                            [
                                {
                                    "id": finalized["id"],
                                    "text": finalized["comment"],
                                    "label": finalized.get("feedback_label", "downvote"),
                                }
                            ]
                        )
        self._save_records(self.pending_path, remaining)
        self._save_records(self.outcomes_path, outcomes)
        return {
            "pending": len(remaining),
            "resolved": len(resolved),
            "outcomes": resolved,
        }

    @property
    def pending_path(self) -> Path:
        return ReviewStrategy.storage_dir(self.repo_path) / "feedback_pending.json"

    @property
    def outcomes_path(self) -> Path:
        return ReviewStrategy.storage_dir(self.repo_path) / "feedback_outcomes.json"

    def _feedback_id(
        self,
        repo_name: str,
        pr_number: int,
        comment: dict,
        posted_at: str,
        index: int,
    ) -> str:
        raw = "|".join(
            [
                repo_name,
                str(pr_number),
                comment.get("file", ""),
                str(comment.get("line_start", comment.get("line", 0)) or 0),
                str(comment.get("line_end", comment.get("line", 0)) or 0),
                comment.get("comment", comment.get("message", "")),
                posted_at,
                str(index),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_records(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _save_records(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
