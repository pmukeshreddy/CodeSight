from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from ares.feedback.strategy import ReviewStrategy


BUG_FIX_PATTERN = re.compile(r"\b(fix|bug|patch|hotfix|revert)\b", re.IGNORECASE)
CRITICAL_PATH_PARTS = ("auth", "security", "payment", "billing", "crypto")
CRITICAL_NAMES = (
    "validate",
    "authenticate",
    "authorize",
    "process_payment",
    "charge",
    "encrypt",
    "decrypt",
    "hash_password",
)
UTILITY_PATH_PARTS = ("utils/", "helpers/", "lib/", "common/")
UTILITY_NAMES = ("format_", "to_string", "convert_", "parse_")
TEST_PATH_PARTS = ("test/", "tests/", "__tests__/", "spec/")
NOISE_PATH_PARTS = ("generated/", "dist/", "build/", "__pycache__/")
NOISE_FILE_NAMES = ("_pb2.py", ".generated.", ".min.js")
LOCKFILES = {"package-lock.json", "yarn.lock", "go.sum", "Cargo.lock"}
INPUT_PARAMS = {"request", "input", "body", "payload", "query", "form_data"}


class NodeClassifier:
    def __init__(
        self,
        G: nx.DiGraph,
        repo_path: str,
        strategy: ReviewStrategy | None = None,
        neo4j_client=None,
        repo_name: str = "",
    ):
        self.G = G
        self.repo_path = os.path.abspath(repo_path)
        self.strategy = strategy or ReviewStrategy()
        # Resolved once — callers never check neo4j.available or repo_name again
        self._neo4j = neo4j_client if (neo4j_client and neo4j_client.available and repo_name) else None
        self._repo = repo_name
        self._git_log_cache: dict[tuple, list[str]] = {}
        self._last_author_cache: dict[tuple, str] = {}
        self._co_changed_cache: dict[str, dict[str, int]] = {}
        self._cache_lock = threading.Lock()

    def classify_all(self, include_git_metadata: bool = False) -> None:
        if include_git_metadata:
            self._add_git_metadata()
        print("[classify] Classifying node risk levels...", file=sys.stderr, flush=True)
        self._classify_risk()
        self.G.graph["git_metadata_mode"] = "full" if include_git_metadata else "lazy"

    def enrich_nodes(self, seed_nodes: list[str], max_hops: int = 1) -> None:
        target_nodes = self._review_neighborhood(seed_nodes, max_hops=max_hops)
        if not target_nodes:
            return
        self._add_git_metadata(target_nodes)
        self._classify_risk(target_nodes)

    def _add_git_metadata(self, node_ids: set[str] | None = None) -> None:
        target_nodes = node_ids or set(self.G.nodes)
        # Filter to function nodes upfront
        func_nodes = [
            n for n in target_nodes
            if n in self.G and self.G.nodes[n].get("type") == "function"
            and not self.G.nodes[n].get("git_metadata_loaded")
        ]
        total_funcs = len(func_nodes)
        if not total_funcs:
            return

        print(
            f"[classify] Adding git metadata for {total_funcs} functions...",
            file=sys.stderr,
            flush=True,
        )

        # Pre-warm co-changed cache (one git log per unique file, parallelized)
        unique_files = {self.G.nodes[n]["file"] for n in func_nodes}
        print(
            f"[classify] Pre-fetching co-change data for {len(unique_files)} files...",
            file=sys.stderr,
            flush=True,
        )
        max_workers = min(16, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(self._get_co_changed_files, unique_files)

        # Collect per-function git data in parallel
        def _fetch_node_metadata(node: str) -> tuple[str, dict]:
            data = self.G.nodes[node]
            filepath = data["file"]
            line_start = data.get("line_start", 0)
            line_end = data.get("line_end", 0)
            log_lines = self._git_log_for_range(filepath, line_start, line_end)
            bug_fix_freq = sum(1 for line in log_lines if BUG_FIX_PATTERN.search(line))
            change_freq = len(log_lines)
            last_author = self._get_last_author(filepath, line_start, line_end)
            co_changed = self._get_co_changed_files(filepath)  # cached from pre-warm
            return node, {
                "bug_fix_freq": bug_fix_freq,
                "change_freq": change_freq,
                "last_author": last_author,
                "co_changed": co_changed,
            }

        log_interval = max(1, total_funcs // 10)
        done_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_node_metadata, n): n for n in func_nodes}
            for future in as_completed(futures):
                node, meta = future.result()
                done_count += 1
                if done_count == 1 or done_count % log_interval == 0 or done_count == total_funcs:
                    print(
                        f"[classify] Git metadata: {done_count}/{total_funcs}",
                        file=sys.stderr,
                        flush=True,
                    )
                self.G.nodes[node]["bug_fix_freq"] = meta["bug_fix_freq"]
                self.G.nodes[node]["change_freq"] = meta["change_freq"]
                self.G.nodes[node]["last_author"] = meta["last_author"]
                for partner, count in meta["co_changed"].items():
                    if partner in self.G:
                        self.G.add_edge(node, partner, type="co_changes", count=count)
                self.G.nodes[node]["git_metadata_loaded"] = True
                if self._neo4j:
                    self._neo4j.update_node_metadata(
                        self._repo,
                        node,
                        {
                            "bug_fix_freq": meta["bug_fix_freq"],
                            "change_freq": meta["change_freq"],
                            "last_author": meta["last_author"],
                            "git_metadata_loaded": True,
                        },
                    )

    def _classify_risk(self, node_ids: set[str] | None = None) -> None:
        target_nodes = node_ids or set(self.G.nodes)
        # Classify file nodes first so _classify_function can use the cached file risk.
        files = []
        functions = []
        others = []
        for node in target_nodes:
            if node not in self.G:
                continue
            node_type = self.G.nodes[node].get("type")
            if node_type == "file":
                files.append(node)
            elif node_type == "function":
                functions.append(node)
            else:
                others.append(node)
        for node in files:
            self.G.nodes[node]["risk"] = self._classify_file(node)
        for node in functions:
            self.G.nodes[node]["risk"] = self._classify_function(node, self.G.nodes[node])
        for node in others:
            self.G.nodes[node]["risk"] = "standard"

    def _review_neighborhood(self, seed_nodes: list[str], max_hops: int = 1) -> set[str]:
        neighborhood: set[str] = set()
        frontier = list(seed_nodes)
        seen = set(frontier)
        for node_id in seed_nodes:
            if node_id in self.G and self.G.nodes[node_id].get("type") == "file":
                neighborhood.update(self._file_functions(node_id))
        while frontier:
            current = frontier.pop()
            if current not in self.G:
                continue
            current_data = self.G.nodes[current]
            if current_data.get("type") == "function":
                neighborhood.add(current)
            current_hops = current_data.get("_review_hops", 0)
            if current_hops >= max_hops:
                continue
            neighbors = list(self.G.predecessors(current)) + list(self.G.successors(current))
            for neighbor in neighbors:
                if neighbor in seen or neighbor not in self.G:
                    continue
                edge = (
                    self.G.edges[neighbor, current]
                    if self.G.has_edge(neighbor, current)
                    else self.G.edges[current, neighbor]
                )
                edge_type = edge.get("type")
                edge_types = set(edge.get("types", []))
                if edge_type:
                    edge_types.add(edge_type)
                if "calls" not in edge_types and "contains" not in edge_types:
                    continue
                if (
                    "contains" in edge_types
                    and current_data.get("type") == "function"
                    and self.G.nodes[neighbor].get("type") == "file"
                ):
                    continue
                seen.add(neighbor)
                self.G.nodes[neighbor]["_review_hops"] = current_hops + 1
                frontier.append(neighbor)
                if self.G.nodes[neighbor].get("type") == "file":
                    neighborhood.update(self._file_functions(neighbor))
        for node_id in seen:
            if node_id in self.G and "_review_hops" in self.G.nodes[node_id]:
                del self.G.nodes[node_id]["_review_hops"]
        return neighborhood

    def _file_functions(self, file_node: str) -> set[str]:
        return {
            target
            for _, target, data in self.G.out_edges(file_node, data=True)
            if data.get("type") == "contains" and self.G.nodes[target].get("type") == "function"
        }

    def _classify_file(self, filepath: str) -> str:
        lowered = filepath.lower()
        filename = os.path.basename(lowered)
        source = self.G.nodes[filepath].get("source", "")
        first_line = source.splitlines()[0].lower() if source.splitlines() else ""
        if any(part in lowered for part in NOISE_PATH_PARTS):
            return "noise"
        if filename in LOCKFILES or any(token in filename for token in NOISE_FILE_NAMES):
            return "noise"
        if "do not edit" in first_line or "auto-generated" in first_line:
            return "noise"
        if any(part in lowered for part in TEST_PATH_PARTS) or filename.startswith("test_") or filename.endswith(
            ("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts")
        ):
            return "test"
        if any(part in lowered for part in CRITICAL_PATH_PARTS):
            return "critical"
        if any(part in lowered for part in UTILITY_PATH_PARTS):
            return "utility"
        return "standard"

    def _classify_function(self, node_id: str, data: dict) -> str:
        file_risk = self.G.nodes[data["file"]].get("risk")
        if not file_risk:
            file_risk = self._classify_file(data["file"])
            self.G.nodes[data["file"]]["risk"] = file_risk
        if file_risk in {"noise", "test"}:
            return file_risk
        name = data.get("name", "").lower()
        filepath = data.get("file", "").lower()
        params = {param.lower() for param in data.get("params", [])}
        if (
            file_risk == "critical"
            or any(part in filepath for part in CRITICAL_PATH_PARTS)
            or any(token in name for token in CRITICAL_NAMES)
            or data.get("bug_fix_freq", 0) >= self.strategy.bug_fix_freq_critical
            or INPUT_PARAMS.intersection(params)
        ):
            return "critical"
        if (
            file_risk == "utility"
            or any(part in filepath for part in UTILITY_PATH_PARTS)
            or any(name.startswith(prefix) for prefix in UTILITY_NAMES)
            or data.get("change_freq", 0) <= self.strategy.change_freq_utility
        ):
            return "utility"
        return "standard"

    def _git_log_for_range(self, filepath: str, start: int, end: int) -> list[str]:
        key = (filepath, start, end)
        with self._cache_lock:
            cached = self._git_log_cache.get(key)
            if cached is not None:
                return cached
        cmd = [
            "git",
            "-C",
            self.repo_path,
            "log",
            "--oneline",
            f"-L{start},{end}:{filepath}",
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            with self._cache_lock:
                self._git_log_cache[key] = []
            return []
        if completed.returncode != 0:
            with self._cache_lock:
                self._git_log_cache[key] = []
            return []
        result = [line for line in completed.stdout.splitlines() if line.strip()]
        with self._cache_lock:
            self._git_log_cache[key] = result
        return result

    def _count_bug_fix_commits(self, filepath: str, start: int, end: int) -> int:
        return sum(1 for line in self._git_log_for_range(filepath, start, end) if BUG_FIX_PATTERN.search(line))

    def _count_total_commits(self, filepath: str, start: int, end: int) -> int:
        return len(self._git_log_for_range(filepath, start, end))

    def _get_last_author(self, filepath: str, start: int, end: int) -> str:
        key = (filepath, start, end)
        with self._cache_lock:
            cached = self._last_author_cache.get(key)
            if cached is not None:
                return cached
        cmd = [
            "git",
            "-C",
            self.repo_path,
            "blame",
            "--line-porcelain",
            f"-L{start},{end}",
            filepath,
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            with self._cache_lock:
                self._last_author_cache[key] = ""
            return ""
        if completed.returncode != 0:
            with self._cache_lock:
                self._last_author_cache[key] = ""
            return ""
        authors = [
            line.split(" ", 1)[1]
            for line in completed.stdout.splitlines()
            if line.startswith("author ")
        ]
        if not authors:
            with self._cache_lock:
                self._last_author_cache[key] = ""
            return ""
        result = Counter(authors).most_common(1)[0][0]
        with self._cache_lock:
            self._last_author_cache[key] = result
        return result

    def _get_co_changed_files(self, filepath: str) -> dict[str, int]:
        with self._cache_lock:
            cached = self._co_changed_cache.get(filepath)
            if cached is not None:
                return cached
        cmd = [
            "git",
            "-C",
            self.repo_path,
            "log",
            "--name-only",
            "--pretty=format:---",
            "--max-count=50",
            "--",
            filepath,
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            with self._cache_lock:
                self._co_changed_cache[filepath] = {}
            return {}
        if completed.returncode != 0:
            with self._cache_lock:
                self._co_changed_cache[filepath] = {}
            return {}
        counts: Counter[str] = Counter()
        for group in completed.stdout.split("---"):
            files_in_commit = {
                line.strip()
                for line in group.strip().splitlines()
                if line.strip()
            }
            for changed in files_in_commit:
                if changed != filepath:
                    counts[changed] += 1
        result = dict(counts.most_common(5))
        with self._cache_lock:
            self._co_changed_cache[filepath] = result
        return result
