from __future__ import annotations

from ares.feedback.strategy import ReviewStrategy
from ares.utils.text_similarity import text_similarity


class Ranker:
    def __init__(self, max_comments: int = 3, strategy: ReviewStrategy | None = None):
        self.max_comments = max_comments
        self.strategy = strategy or ReviewStrategy()

    def rank_and_cap(self, static_findings: list[dict], verified_comments: list[dict]) -> list[dict]:
        static_stream = [{**finding, "source": "static"} for finding in static_findings]
        llm_stream = [{**comment, "source": "llm"} for comment in verified_comments]
        merged = self._deduplicate(static_stream, llm_stream)
        merged = [comment for comment in merged if self._passes_min_confidence(comment)]
        merged.sort(key=self._priority_key)
        final = merged[: self.max_comments]
        for comment in final:
            comment["github_body"] = self._format_for_github(comment)
        return final

    def _deduplicate(self, static_findings: list[dict], verified_comments: list[dict]) -> list[dict]:
        final = list(verified_comments)
        for finding in static_findings:
            duplicate = self._find_duplicate(finding, final)
            if duplicate is None:
                final.append(finding)
                continue
            duplicate["confirmed_by_static"] = True
            duplicate.setdefault("rule_ids", []).append(finding.get("rule_id"))
        return final

    def _find_duplicate(self, finding: dict, comments: list[dict]) -> dict | None:
        for comment in comments:
            same_file = finding.get("file") == comment.get("file")
            overlapping = not (
                finding.get("line_end", 0) < comment.get("line_start", 0)
                or finding.get("line_start", 0) > comment.get("line_end", 0)
            )
            similar_text = text_similarity(
                finding.get("message") or "",
                comment.get("comment") or "",
            ) > 0.45
            if same_file and overlapping and similar_text:
                return comment
        return None

    def _priority_key(self, comment: dict) -> tuple:
        severity = str(comment.get("severity", "warning")).lower()
        weight = float(self.strategy.severity_weights.get(severity, 0.5))
        tool_weight = 0
        if comment.get("tool") == "semgrep":
            tool_weight = 2
        elif comment.get("tool") == "ruff":
            tool_weight = 1
        return (
            -weight,
            -float(comment.get("confidence", 0.0)),
            -int(bool(comment.get("confirmed_by_static", False))),
            -tool_weight,
            -int(comment.get("source") == "static"),
            comment.get("line_start", 0),
        )

    def _passes_min_confidence(self, comment: dict) -> bool:
        if comment.get("source") == "static":
            return True
        return float(comment.get("confidence", 0.0)) >= float(self.strategy.min_confidence)

    def _format_for_github(self, comment: dict) -> str:
        severity = comment.get("severity", "warning").upper()
        body = f"**{severity}**: {comment.get('comment') or comment.get('message', '')}"
        if comment.get("confirmed_by_static"):
            body += "\n\nConfirmed by static analysis."
        verification_passed = comment.get("verification", {}).get("status") == "passed"
        suggested_code = comment.get("suggested_code", "").rstrip()
        if verification_passed and suggested_code:
            body += f"\n\n```suggestion\n{suggested_code}\n```"
        elif verification_passed and comment.get("fix_diff"):
            body += f"\n\n```diff\n{comment['fix_diff'].rstrip()}\n```"
        return body
