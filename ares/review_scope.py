from __future__ import annotations

import os
import re


REVIEWABLE_SOURCE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx"}
TEST_PATH_SEGMENTS = {"test", "tests", "__tests__", "spec"}
NOISE_PATH_SEGMENTS = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    "generated",
    "node_modules",
    "vendor",
}
LOCKFILES = {"package-lock.json", "yarn.lock", "go.sum", "cargo.lock", "poetry.lock", "pdm.lock"}
NOISE_FILE_TOKENS = ("_pb2.py", ".generated.", ".min.js")
TEST_FILE_SUFFIXES = ("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts")
MAINTENANCE_PATTERN = re.compile(
    r"\b(chore|deps?|dependency|dependabot|lockfile|bump|upgrade|ci only|lint only|linting only|formatting only)\b",
    re.IGNORECASE,
)


def is_reviewable_source_file(path: str) -> bool:
    lowered = (path or "").strip().lower()
    if not lowered:
        return False
    filename = os.path.basename(lowered)
    if filename in LOCKFILES or any(token in filename for token in NOISE_FILE_TOKENS):
        return False
    if filename.startswith("test_") or filename.endswith(TEST_FILE_SUFFIXES):
        return False
    parts = [part for part in lowered.split("/") if part]
    if any(part in TEST_PATH_SEGMENTS for part in parts):
        return False
    if any(part in NOISE_PATH_SEGMENTS for part in parts):
        return False
    _, ext = os.path.splitext(filename)
    return ext in REVIEWABLE_SOURCE_EXTENSIONS


def reviewable_source_files(paths: list[str]) -> list[str]:
    return [path for path in paths if is_reviewable_source_file(path)]


def looks_like_maintenance_pr_text(text: str) -> bool:
    return bool(MAINTENANCE_PATTERN.search(text or ""))


def is_maintenance_only_pr(text: str, changed_files: list[str]) -> bool:
    return looks_like_maintenance_pr_text(text) and not reviewable_source_files(changed_files)
