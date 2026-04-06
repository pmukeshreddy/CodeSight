from __future__ import annotations

import os
import re
from collections import defaultdict

from ares.review_scope import is_maintenance_only_pr, reviewable_source_files


class Investigator:
    def __init__(self, graph_query, repo_path: str):
        self.graph_query = graph_query
        self.repo_path = os.path.abspath(repo_path)

    def investigate(
        self,
        pr_diff: str,
        pr_description: str,
        changed_nodes: list[str],
        changed_files: list[str] | None = None,
        reviewable_changed_files: list[str] | None = None,
        diff_hunks: dict[str, list[dict]] | None = None,
    ) -> list[dict]:
        changed_files = changed_files or []
        reviewable_changed_files = reviewable_changed_files or reviewable_source_files(changed_files)
        if not reviewable_changed_files:
            return []
        intent = self._extract_pr_intent(pr_description, changed_files)
        if intent.get("maintenance_only"):
            return []
        if diff_hunks is None:
            diff_hunks = self._parse_diff_hunks(pr_diff)
        targets = self.graph_query.get_review_targets(changed_nodes, diff_hunks=diff_hunks)
        if not targets and changed_nodes:
            targets = self.graph_query.get_fallback_targets(changed_nodes, diff_hunks=diff_hunks, limit=3)
        if not targets and reviewable_changed_files:
            targets = self.graph_query.get_file_fallback_targets(reviewable_changed_files, diff_hunks=diff_hunks)
        investigated = []
        for target in targets:
            context = target["context"]
            investigated.append(
                {
                    "node_id": target["node_id"],
                    "file": target["file"],
                    "risk": target.get("risk", "standard"),
                    "original_risk": target.get("original_risk", target.get("risk", "standard")),
                    "caller_count": target.get("caller_count", 0),
                    "change_type": target.get("change_type", "unknown"),
                    "line_start": target["line_start"],
                    "line_end": target["line_end"],
                    "function_source": target["source"],
                    "function_signature": target["signature"],
                    "diff_hunk": self._diff_for_target(target["file"], target["line_start"], target["line_end"], diff_hunks),
                    "pr_intent": self._format_intent(intent),
                    "review_instruction": self._instruction_for_intent(intent),
                    "context": {
                        "callers": self._format_context_block(context["callers"]),
                        "callees": self._format_context_block(context["callees"]),
                        "tests": self._format_context_block(context["tests"]),
                        "bug_history": context["bug_history"],
                        "co_changes": context["co_changes"],
                    },
                    "test_files": [
                        item["file"] for item in context["tests"] if item.get("file")
                    ],
                }
            )
        return investigated

    def _extract_pr_intent(self, pr_description: str, changed_files: list[str]) -> dict:
        text = (pr_description or "").lower()
        scope_match = re.search(
            r"\b(auth|security|payment|billing|profile|user|api|cache|db|database|query|frontend|ui)\b",
            text,
        )
        maintenance_only = is_maintenance_only_pr(pr_description, changed_files)
        if re.search(r"\b(refactor|clean up|cleanup|reorganize|rename)\b", text):
            intent = "refactor"
        elif re.search(r"\b(fix|bug|patch|resolve|hotfix)\b", text):
            intent = "bugfix"
        elif re.search(r"\b(add|implement|introduce|new feature|feature)\b", text):
            intent = "feature"
        elif maintenance_only:
            intent = "chore"
        else:
            intent = "unknown"
        stated_behavior_change = not re.search(
            r"\b(no behavior change|refactor only|non-functional|internal only)\b",
            text,
        )
        return {
            "intent": intent,
            "scope": scope_match.group(1) if scope_match else "unknown",
            "stated_behavior_change": stated_behavior_change,
            "maintenance_only": maintenance_only,
        }

    def _instruction_for_intent(self, intent: dict) -> str:
        kind = intent["intent"]
        if kind == "refactor":
            return "Flag behavior changes only. Ignore style, naming, and formatting."
        if kind == "bugfix":
            return "Verify the fix is correct and look for regressions or incomplete edge-case handling."
        if kind == "feature":
            return "Check for correctness bugs, missing error handling, and security issues."
        return "Only flag concrete bugs or security issues. Ignore style and naming."

    def _format_intent(self, intent: dict) -> str:
        behavior = "behavior change expected" if intent["stated_behavior_change"] else "no behavior change expected"
        return f"{intent['intent']} in {intent['scope']} ({behavior})"

    def _format_context_block(self, items: list[dict]) -> str:
        if not items:
            return "None"
        blocks = []
        for item in items:
            header = f"{item['node_id']} [{item['file']}:{item['line_start']}-{item['line_end']}]"
            blocks.append(f"{header}\n{item['content']}".strip())
        return "\n\n".join(blocks)

    def _parse_diff_hunks(self, diff_text: str) -> dict[str, list[dict]]:
        file_hunks: dict[str, list[dict]] = defaultdict(list)
        current_file = ""
        current_hunk: dict | None = None
        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
                current_hunk = None
                continue
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if not current_file or not match:
                    continue
                start = int(match.group(1))
                count = int(match.group(2) or "1")
                current_hunk = {
                    "start": start,
                    "end": start + max(count - 1, 0),
                    "lines": [line],
                }
                file_hunks[current_file].append(current_hunk)
                continue
            if current_hunk is not None:
                current_hunk["lines"].append(line)
        return file_hunks

    def _diff_for_target(
        self, filepath: str, line_start: int, line_end: int, diff_hunks: dict[str, list[dict]]
    ) -> str:
        hunks = []
        for hunk in diff_hunks.get(filepath, []):
            overlaps = not (line_end < hunk["start"] or line_start > hunk["end"])
            if overlaps:
                hunks.append("\n".join(hunk["lines"]))
        return "\n\n".join(hunks) if hunks else "No matching diff hunk found."
