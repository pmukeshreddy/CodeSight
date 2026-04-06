from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


SEMGRP_INTERESTING = (
    "security",
    "sql",
    "xss",
    "ssrf",
    "traversal",
    "correctness",
    "null",
    "type",
)


class StaticAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)

    def analyze_changed_files(self, changed_files: list[str]) -> list[dict]:
        files = [path for path in changed_files if (Path(self.repo_path) / path).exists()]
        findings: list[dict] = []
        findings.extend(self._run_ruff(files))
        findings.extend(self._run_semgrep(files))
        return findings

    def _run_ruff(self, files: list[str]) -> list[dict]:
        if not files:
            return []
        try:
            completed = subprocess.run(
                ["ruff", "check", "--output-format", "json", *files],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return []
        if completed.returncode not in {0, 1}:
            return []
        raw_findings = json.loads(completed.stdout or "[]")
        findings: list[dict] = []
        for finding in raw_findings:
            code = (finding.get("code") or "").upper()
            if not code.startswith(("E", "F")):
                continue
            location = finding.get("location", {})
            fix = finding.get("fix") or {}
            filename = finding.get("filename", "")
            if os.path.isabs(filename):
                filename = os.path.relpath(filename, self.repo_path)
            findings.append(
                {
                    "source": "static",
                    "tool": "ruff",
                    "file": filename,
                    "line_start": location.get("row", 1),
                    "line_end": location.get("row", 1),
                    "column": location.get("column", 1),
                    "rule_id": code,
                    "message": finding.get("message", ""),
                    "severity": "high" if code.startswith(("E", "F")) else "medium",
                    "suggested_fix": fix.get("message") or "",
                    "suggested_code": self._extract_ruff_replacement(fix),
                }
            )
        return findings

    def _extract_ruff_replacement(self, fix: dict) -> str:
        edits = fix.get("edits") or []
        if not edits:
            return ""
        replacement = edits[0].get("content")
        return replacement or ""

    def _run_semgrep(self, files: list[str]) -> list[dict]:
        if not files:
            return []
        try:
            completed = subprocess.run(
                ["semgrep", "--config", "auto", "--json", "--quiet", *files],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return []
        if completed.returncode not in {0, 1}:
            return []
        payload = json.loads(completed.stdout or '{"results": []}')
        findings: list[dict] = []
        for result in payload.get("results", []):
            extra = result.get("extra", {})
            metadata = extra.get("metadata", {})
            severity = (extra.get("severity") or "WARNING").upper()
            joined = " ".join(
                [
                    result.get("check_id", ""),
                    extra.get("message", ""),
                    json.dumps(metadata, sort_keys=True),
                ]
            ).lower()
            if not any(token in joined for token in SEMGRP_INTERESTING):
                continue
            start = result.get("start", {})
            end = result.get("end", {})
            fix = extra.get("fix") or ""
            filepath = result.get("path", "")
            if os.path.isabs(filepath):
                filepath = os.path.relpath(filepath, self.repo_path)
            findings.append(
                {
                    "source": "static",
                    "tool": "semgrep",
                    "file": filepath,
                    "line_start": start.get("line", 1),
                    "line_end": end.get("line", start.get("line", 1)),
                    "column": start.get("col", 1),
                    "rule_id": result.get("check_id", ""),
                    "message": extra.get("message", ""),
                    "severity": "high" if severity in {"ERROR", "WARNING"} else "medium",
                    "suggested_fix": fix,
                    "suggested_code": fix,
                }
            )
        return findings
