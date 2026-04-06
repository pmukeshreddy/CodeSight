from __future__ import annotations

import ast
import copy
import os
import re
from collections import deque
from typing import Any

import networkx as nx

from ares.feedback.strategy import ReviewStrategy

try:
    from tree_sitter import Language, Parser
except ImportError:  # pragma: no cover - optional dependency
    Language = None
    Parser = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_javascript as tsjavascript
except ImportError:  # pragma: no cover - optional dependency
    tsjavascript = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_python as tspython
except ImportError:  # pragma: no cover - optional dependency
    tspython = None

try:  # pragma: no cover - optional dependency
    import tree_sitter_typescript as tstypescript
except ImportError:  # pragma: no cover - optional dependency
    tstypescript = None

class GraphQuery:
    def __init__(
        self,
        G: nx.DiGraph,
        strategy: ReviewStrategy | None = None,
        neo4j_client=None,
        repo_name: str = "",
    ):
        self.G = G
        self.strategy = strategy or ReviewStrategy()
        self._parsers = self._build_parsers()
        # When Neo4j is available all graph traversal (callers, callees,
        # transitive counts) runs as Cypher inside Neo4j — not networkx BFS.
        self._neo4j = neo4j_client if (neo4j_client and neo4j_client.available) else None
        self._repo = repo_name
        self._file_index: dict[str, list[str]] = self._build_file_index()
        self._caller_count_cache: dict[str, int] = {}
        self._direct_caller_count_cache: dict[str, int] = {}

    def _build_file_index(self) -> dict[str, list[str]]:
        """Pre-compute file -> [function node_id] mapping for O(1) lookups."""
        index: dict[str, list[tuple[int, str]]] = {}
        for node_id, data in self.G.nodes(data=True):
            if data.get("type") != "function":
                continue
            filepath = data.get("file", "")
            if filepath:
                index.setdefault(filepath, []).append(
                    (int(data.get("line_start", 0) or 0), node_id)
                )
        return {
            filepath: [node_id for _, node_id in sorted(entries)]
            for filepath, entries in index.items()
        }

    def get_review_targets(
        self,
        changed_nodes: list[str],
        diff_hunks: dict[str, list[dict]] | None = None,
        max_targets: int = 15,
    ) -> list[dict]:
        targets = []
        seen: set[str] = set()
        for node_id in changed_nodes:
            if node_id in seen:
                continue
            seen.add(node_id)
            target = self._build_target(node_id, diff_hunks=diff_hunks)
            if target is not None:
                targets.append(target)
        targets.sort(
            key=lambda target: (
                0 if target["risk"] == "critical" else 1,
                -target["bug_fix_freq"],
                -target["change_freq"],
            )
        )
        # Cap to avoid excessive LLM calls on large PRs.
        # Critical targets are always at the front due to sorting.
        if len(targets) > max_targets:
            import sys
            print(
                f"[review_targets] Capped {len(targets)} targets to {max_targets} "
                f"(prioritized by risk and change frequency)",
                file=sys.stderr, flush=True,
            )
        return targets[:max_targets]

    def get_fallback_targets(
        self,
        changed_nodes: list[str],
        diff_hunks: dict[str, list[dict]] | None = None,
        limit: int = 3,
    ) -> list[dict]:
        targets = []
        for node_id in changed_nodes:
            if len(targets) >= limit:
                break
            target = self._build_target(node_id, diff_hunks=diff_hunks, force=True)
            if target is None:
                continue
            target["fallback_target"] = True
            targets.append(target)
        return targets

    def get_file_fallback_targets(
        self,
        filepaths: list[str],
        diff_hunks: dict[str, list[dict]] | None = None,
        limit_per_file: int = 3,
    ) -> list[dict]:
        targets = []
        seen: set[str] = set()
        for filepath in filepaths:
            if len(targets) >= max(1, len(filepaths) * limit_per_file):
                break
            functions = self._functions_for_file(filepath)
            for node_id in functions[:limit_per_file]:
                if node_id in seen:
                    continue
                seen.add(node_id)
                target = self._build_target(node_id, diff_hunks=diff_hunks, force=True)
                if target is None:
                    continue
                target["fallback_target"] = True
                targets.append(target)
        return targets

    def _build_target(
        self,
        node_id: str,
        diff_hunks: dict[str, list[dict]] | None = None,
        force: bool = False,
    ) -> dict | None:
        if node_id not in self.G:
            return None
        node_data = self.G.nodes[node_id]
        risk = node_data.get("risk", "standard")
        if risk in {"noise", "test"}:
            return None
        change_type = self._change_type_for_node(node_id, diff_hunks)
        caller_count = self._transitive_caller_count(node_id)
        if risk == "utility" and not force and not self._should_review_utility(caller_count, change_type):
            return None
        direct_caller_count = self._direct_caller_count(node_id)
        effective_risk = "standard" if risk == "utility" else risk
        full_source = node_data.get("source", "")
        trimmed = self._trim_source_to_diff(
            full_source,
            node_data.get("line_start", 1),
            node_data.get("file", ""),
            diff_hunks,
        )
        return {
            "node_id": node_id,
            "risk": effective_risk,
            "original_risk": risk,
            "direct_caller_count": direct_caller_count,
            "caller_count": caller_count,
            "change_type": change_type,
            "bug_fix_freq": node_data.get("bug_fix_freq", 0),
            "change_freq": node_data.get("change_freq", 0),
            "source": trimmed,
            "context": self._assemble_context(node_id),
            "file": node_data.get("file", node_id),
            "line_start": node_data.get("line_start", 1),
            "line_end": node_data.get("line_end", 1),
            "signature": node_data.get("signature", ""),
        }

    def _trim_source_to_diff(
        self,
        full_source: str,
        func_line_start: int,
        filepath: str,
        diff_hunks: dict[str, list[dict]] | None,
        context_lines: int = 10,
        max_lines: int = 60,
    ) -> str:
        """Return a trimmed version of the function source, keeping only lines
        near diff hunks plus surrounding context. Functions shorter than
        *max_lines* are returned as-is."""
        lines = full_source.splitlines()
        if len(lines) <= max_lines or not diff_hunks:
            return full_source
        hunks = diff_hunks.get(filepath, [])
        if not hunks:
            # No hunks for this file — return signature + first/last few lines
            return "\n".join(lines[:max_lines])
        # Convert hunk line numbers (file-absolute) to function-relative
        keep = set()
        # Always keep the signature (first 2 lines)
        keep.update(range(0, min(2, len(lines))))
        for hunk in hunks:
            rel_start = hunk["start"] - func_line_start
            rel_end = hunk["end"] - func_line_start
            for i in range(
                max(0, rel_start - context_lines),
                min(len(lines), rel_end + context_lines + 1),
            ):
                keep.add(i)
        # Always keep last line (closing bracket/return)
        if lines:
            keep.add(len(lines) - 1)
        kept = sorted(keep)
        result = []
        prev = -1
        for i in kept:
            if prev >= 0 and i > prev + 1:
                result.append(f"    # ... ({i - prev - 1} lines omitted)")
            result.append(lines[i])
            prev = i
        return "\n".join(result)

    def _should_review_utility(self, caller_count: int, change_type: str) -> bool:
        if caller_count < self.strategy.utility_blast_radius_threshold:
            return False
        return change_type not in {"cosmetic", "rename"}

    def _functions_for_file(self, filepath: str) -> list[str]:
        return list(self._file_index.get(filepath, []))

    def _direct_caller_count(self, node_id: str) -> int:
        if node_id in self._direct_caller_count_cache:
            return self._direct_caller_count_cache[node_id]
        if self._neo4j:
            count = self._neo4j.query_direct_caller_count(self._repo, node_id)
        else:
            count = sum(
                1 for predecessor in self.G.predecessors(node_id)
                if self._edge_has_type(self.G.edges[predecessor, node_id], {"calls"})
            )
        self._direct_caller_count_cache[node_id] = count
        return count

    def _transitive_caller_count(self, node_id: str) -> int:
        if node_id in self._caller_count_cache:
            return self._caller_count_cache[node_id]
        if self._neo4j:
            count = self._neo4j.query_transitive_caller_count(self._repo, node_id)
        else:
            bfs_queue = deque([node_id])
            seen = {node_id}
            count = 0
            while bfs_queue:
                current = bfs_queue.popleft()
                for predecessor in self.G.predecessors(current):
                    if predecessor in seen:
                        continue
                    if not self._edge_has_type(self.G.edges[predecessor, current], {"calls"}):
                        continue
                    seen.add(predecessor)
                    count += 1
                    bfs_queue.append(predecessor)
        self._caller_count_cache[node_id] = count
        return count

    def _change_type_for_node(
        self,
        node_id: str,
        diff_hunks: dict[str, list[dict]] | None,
    ) -> str:
        if not diff_hunks:
            return "unknown"
        node_data = self.G.nodes[node_id]
        source = node_data.get("source", "")
        filepath = node_data.get("file", "")
        if not source or not filepath:
            return "unknown"
        overlapping_hunks = self._overlapping_hunks(
            filepath,
            node_data.get("line_start", 1),
            node_data.get("line_end", 1),
            diff_hunks,
        )
        if not overlapping_hunks:
            return "unknown"
        old_source = self._reconstruct_old_node_source(
            source,
            node_data.get("line_start", 1),
            node_data.get("line_end", 1),
            overlapping_hunks,
        )
        if old_source is None:
            return "unknown"
        diff = self.get_ast_diff(
            old_source,
            source,
            self._language_for_file(filepath),
        )
        return diff.get("change_type", "unknown")

    def _overlapping_hunks(
        self,
        filepath: str,
        line_start: int,
        line_end: int,
        diff_hunks: dict[str, list[dict]],
    ) -> list[dict]:
        overlaps = []
        for hunk in diff_hunks.get(filepath, []):
            if not (line_end < hunk["start"] or line_start > hunk["end"]):
                overlaps.append(hunk)
        return overlaps

    def _reconstruct_old_node_source(
        self,
        new_source: str,
        node_line_start: int,
        node_line_end: int,
        overlapping_hunks: list[dict],
    ) -> str | None:
        new_lines = new_source.splitlines()
        old_lines = list(new_lines)
        segments = []
        for hunk in overlapping_hunks:
            segment = self._old_new_segments_for_hunk(hunk, node_line_start, node_line_end)
            if segment is not None:
                segments.append(segment)
        if not segments:
            return None
        for start_idx, new_count, old_segment in sorted(segments, key=lambda item: item[0], reverse=True):
            old_lines[start_idx : start_idx + new_count] = old_segment
        trailing_newline = "\n" if new_source.endswith("\n") else ""
        return "\n".join(old_lines) + trailing_newline

    def _old_new_segments_for_hunk(
        self,
        hunk: dict,
        node_line_start: int,
        node_line_end: int,
    ) -> tuple[int, int, list[str]] | None:
        old_line, new_line = self._hunk_line_numbers(hunk.get("lines", []))
        if old_line is None or new_line is None:
            return None
        segment_start_abs: int | None = None
        old_segment: list[str] = []
        new_segment_count = 0
        for raw_line in hunk.get("lines", [])[1:]:
            if not raw_line or raw_line.startswith("\\"):
                continue
            marker = raw_line[0]
            content = raw_line[1:]
            if marker == " ":
                if node_line_start <= new_line <= node_line_end:
                    if segment_start_abs is None:
                        segment_start_abs = new_line
                    old_segment.append(content)
                    new_segment_count += 1
                old_line += 1
                new_line += 1
                continue
            if marker == "+":
                if node_line_start <= new_line <= node_line_end:
                    if segment_start_abs is None:
                        segment_start_abs = new_line
                    new_segment_count += 1
                new_line += 1
                continue
            if marker == "-":
                if node_line_start <= new_line <= node_line_end + 1:
                    if segment_start_abs is None:
                        segment_start_abs = min(new_line, node_line_end + 1)
                    old_segment.append(content)
                old_line += 1
        if segment_start_abs is None:
            return None
        return segment_start_abs - node_line_start, new_segment_count, old_segment

    def _hunk_line_numbers(self, lines: list[str]) -> tuple[int | None, int | None]:
        if not lines:
            return None, None
        match = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", lines[0])
        if match is None:
            return None, None
        return int(match.group(1)), int(match.group(2))

    def _language_for_file(self, filepath: str) -> str:
        extension = os.path.splitext(filepath)[1].lower()
        if extension == ".py":
            return "python"
        if extension in {".ts", ".tsx"}:
            return "typescript"
        return "javascript"

    def _assemble_context(self, node_id: str, max_hops: int = 2, max_context_nodes: int = 5) -> dict:
        callers = self._bfs_related(node_id, direction="up", edge_types={"calls"}, max_hops=max_hops)[:max_context_nodes]
        callees = self._bfs_related(node_id, direction="down", edge_types={"calls"}, max_hops=max_hops)[:max_context_nodes]
        tests = self._collect_neighbors(node_id, edge_type="tests")[:3]
        co_changes = self._collect_neighbors(node_id, edge_type="co_changes")[:3]
        node_data = self.G.nodes[node_id]
        bug_history = (
            f"{node_data.get('bug_fix_freq', 0)} bug-fix commits, "
            f"{node_data.get('change_freq', 0)} total changes, "
            f"last author: {node_data.get('last_author', 'unknown') or 'unknown'}"
        )
        return {
            "callers": [self._summarize_node(related_id, hop) for related_id, hop in callers],
            "callees": [self._summarize_node(related_id, hop) for related_id, hop in callees],
            "tests": [self._summarize_node(related_id, 1) for related_id in tests],
            "co_changes": [
                {
                    "node_id": related_id,
                    "file": self.G.nodes[related_id].get("file", related_id),
                    "count": self.G.edges[node_id, related_id].get("count", 0),
                }
                for related_id in co_changes
            ],
            "bug_history": bug_history,
        }

    def _bfs_related(
        self, node_id: str, direction: str, edge_types: set[str], max_hops: int
    ) -> list[tuple[str, int]]:
        # Delegate to Cypher when Neo4j available — runs inside the graph engine
        if self._neo4j and "calls" in edge_types:
            if direction == "up":
                rows = self._neo4j.query_callers(self._repo, node_id, max_hops)
            else:
                rows = self._neo4j.query_callees(self._repo, node_id, max_hops)
            return [(r["local_id"], r["hop"]) for r in rows if r.get("local_id")]

        # Fallback: networkx BFS
        queue = deque([(node_id, 0)])
        seen = {node_id}
        results: list[tuple[str, int]] = []
        while queue:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            neighbors = (
                self.G.predecessors(current) if direction == "up" else self.G.successors(current)
            )
            for neighbor in neighbors:
                edge = self.G.edges[neighbor, current] if direction == "up" else self.G.edges[current, neighbor]
                if not self._edge_has_type(edge, edge_types) or neighbor in seen:
                    continue
                seen.add(neighbor)
                results.append((neighbor, hop + 1))
                queue.append((neighbor, hop + 1))
        return results

    def _collect_neighbors(self, node_id: str, edge_type: str) -> list[str]:
        if self._neo4j:
            if edge_type == "tests":
                return [r["local_id"] for r in self._neo4j.query_tests(self._repo, node_id) if r.get("local_id")]
            if edge_type == "co_changes":
                return [r["local_id"] for r in self._neo4j.query_co_changes(self._repo, node_id) if r.get("local_id")]

        # Fallback: networkx
        related: list[str] = []
        for source, target, data in self.G.in_edges(node_id, data=True):
            if self._edge_has_type(data, {edge_type}):
                related.append(source)
        for source, target, data in self.G.out_edges(node_id, data=True):
            if self._edge_has_type(data, {edge_type}):
                related.append(target)
        return list(dict.fromkeys(related))

    def _edge_has_type(self, edge: dict, edge_types: set[str]) -> bool:
        types = set(edge.get("types", []))
        if edge.get("type"):
            types.add(edge["type"])
        return bool(types.intersection(edge_types))

    def _summarize_node(self, node_id: str, hop: int) -> dict:
        data = self.G.nodes[node_id]
        source = data.get("source", "")
        # Always send just the signature for context nodes — full source
        # was sending 100-500 line functions for every caller/callee,
        # exploding token costs.
        signature = data.get("signature", "")
        if not signature and source:
            signature = source.splitlines()[0]
        content = signature
        return {
            "node_id": node_id,
            "file": data.get("file", node_id),
            "hop": hop,
            "type": data.get("type", "function"),
            "risk": data.get("risk", "standard"),
            "line_start": data.get("line_start", 1),
            "line_end": data.get("line_end", 1),
            "content": content,
        }

    def get_ast_diff(self, old_source: bytes | str, new_source: bytes | str, language: str) -> dict:
        if language == "python":
            old_str = old_source if isinstance(old_source, str) else old_source.decode("utf-8")
            new_str = new_source if isinstance(new_source, str) else new_source.decode("utf-8")
            return self._python_ast_diff(old_str, new_str)
        old_bytes = old_source.encode("utf-8") if isinstance(old_source, str) else old_source
        new_bytes = new_source.encode("utf-8") if isinstance(new_source, str) else new_source
        tree_old = self._parse_tree(language, old_bytes)
        tree_new = self._parse_tree(language, new_bytes)
        if tree_old is None or tree_new is None:
            structural_change = old_bytes != new_bytes
            return {
                "structural_change": structural_change,
                "changed_nodes": [],
                "change_type": "logic" if structural_change else "cosmetic",
            }
        full_old = self._tree_signature(tree_old.root_node, old_bytes, normalize_identifiers=False)
        full_new = self._tree_signature(tree_new.root_node, new_bytes, normalize_identifiers=False)
        if full_old == full_new:
            return {"structural_change": False, "changed_nodes": [], "change_type": "cosmetic"}
        normalized_old = self._tree_signature(
            tree_old.root_node, old_bytes, normalize_identifiers=True
        )
        normalized_new = self._tree_signature(
            tree_new.root_node, new_bytes, normalize_identifiers=True
        )
        change_type = "rename" if normalized_old == normalized_new else "logic"
        changed_nodes = self._top_level_named_children(tree_old.root_node, old_bytes) ^ self._top_level_named_children(
            tree_new.root_node, new_bytes
        )
        return {
            "structural_change": change_type == "logic",
            "changed_nodes": sorted(changed_nodes),
            "change_type": change_type,
        }

    def _build_parsers(self) -> dict[str, Any]:
        if Parser is None or Language is None:
            return {}
        parsers: dict[str, Any] = {}
        if tsjavascript is not None:
            parsers["javascript"] = self._get_parser(tsjavascript.language())
        if tstypescript is not None:
            parsers["typescript"] = self._get_parser(tstypescript.language_typescript())
        if tspython is not None:
            parsers["python"] = self._get_parser(tspython.language())
        return parsers

    def _get_parser(self, language: Any) -> Parser:
        parser = Parser()
        parser.language = Language(language)
        return parser

    def _parse_tree(self, language: str, source: bytes):
        parser = self._parsers.get(language)
        if parser is None:
            return None
        return parser.parse(source)

    def _python_ast_diff(self, old_source: str, new_source: str) -> dict:
        try:
            old_tree = ast.parse(old_source)
            new_tree = ast.parse(new_source)
        except SyntaxError:
            structural_change = old_source != new_source
            return {
                "structural_change": structural_change,
                "changed_nodes": [],
                "change_type": "logic" if structural_change else "cosmetic",
            }
        full_old = ast.dump(old_tree, include_attributes=False)
        full_new = ast.dump(new_tree, include_attributes=False)
        if full_old == full_new:
            return {"structural_change": False, "changed_nodes": [], "change_type": "cosmetic"}
        # Reuse already-parsed trees for normalization instead of re-parsing.
        normalized_old = ast.dump(_NameNormalizer().visit(copy.deepcopy(old_tree)), include_attributes=False)
        normalized_new = ast.dump(_NameNormalizer().visit(copy.deepcopy(new_tree)), include_attributes=False)
        change_type = "rename" if normalized_old == normalized_new else "logic"
        changed_nodes = self._python_top_level_names(old_tree) ^ self._python_top_level_names(new_tree)
        return {
            "structural_change": change_type == "logic",
            "changed_nodes": sorted(changed_nodes),
            "change_type": change_type,
        }

    def _python_top_level_names(self, tree: ast.AST) -> set[str]:
        names = set()
        for child in getattr(tree, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
        return names

    def _tree_signature(self, node: Any, source: bytes, normalize_identifiers: bool) -> tuple:
        if not getattr(node, "is_named", True):
            return ()
        token = node.type
        if normalize_identifiers and token in {"identifier", "property_identifier", "type_identifier"}:
            value = "<id>"
        else:
            value = source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
            if token not in {"identifier", "property_identifier", "type_identifier", "number", "string"}:
                value = ""
        children = tuple(
            child_sig
            for child in node.children
            for child_sig in [self._tree_signature(child, source, normalize_identifiers)]
            if child_sig
        )
        return (token, value, children)

    def _top_level_named_children(self, node: Any, source: bytes) -> set[str]:
        names = set()
        for child in node.named_children:
            snippet = source[child.start_byte : child.end_byte].decode("utf-8", errors="ignore")
            first_line = snippet.splitlines()[0].strip() if snippet.splitlines() else child.type
            names.add(f"{child.type}:{first_line}")
        return names


class _NameNormalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "__func__"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "__func__"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = "__class__"
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = "__name__"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "__arg__"
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = "__attr__"
        return node
