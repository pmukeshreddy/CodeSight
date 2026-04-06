from __future__ import annotations

import hashlib
import os
from pathlib import Path

import networkx as nx

from ares.feedback.strategy import ReviewStrategy
from ares.graph.classifier import NodeClassifier
from ares.graph.parser import RepoParser, SUPPORTED_EXTENSIONS


class RepositoryIndexer:
    def __init__(
        self,
        repo_path: str,
        output_dir: str | None = None,
        strategy: ReviewStrategy | None = None,
        neo4j_client=None,
        repo_name: str = "",
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.output_dir = output_dir or str(Path(self.repo_path) / ".ares")
        self.strategy = strategy or ReviewStrategy()
        self._neo4j = neo4j_client if (neo4j_client and neo4j_client.available and repo_name) else None
        self._repo = repo_name

    def build(self, include_git_metadata: bool = False) -> nx.DiGraph:
        parser = RepoParser(self.repo_path)
        graph = parser.parse_repo()
        NodeClassifier(
            graph,
            self.repo_path,
            strategy=self.strategy,
            neo4j_client=self._neo4j,
            repo_name=self._repo,
        ).classify_all(include_git_metadata=include_git_metadata)
        return graph

    def save(self, graph: nx.DiGraph) -> None:
        if not self._neo4j:
            raise RuntimeError(
                "Neo4j is required for graph storage. Set ARES_NEO4J_URI, "
                "ARES_NEO4J_USER, and ARES_NEO4J_PASSWORD environment variables."
            )
        self._neo4j.save_graph(self._repo, graph)

    def load(self) -> nx.DiGraph:
        if not self._neo4j:
            raise RuntimeError(
                "Neo4j is required for graph storage. Set ARES_NEO4J_URI, "
                "ARES_NEO4J_USER, and ARES_NEO4J_PASSWORD environment variables."
            )
        graph = self._neo4j.load_graph(self._repo)
        if graph is not None:
            return graph
        raise FileNotFoundError(
            f"No graph found in Neo4j for repo '{self._repo}'. "
            "Run 'index' first to build and store the graph."
        )

    def build_and_save(self, include_git_metadata: bool = False) -> None:
        # Neo4j persists git metadata — computing it on first build is free
        # for all future PRs (no git log re-run). Always enable when Neo4j is up.
        graph = self.build(include_git_metadata=include_git_metadata or bool(self._neo4j))
        self.save(graph)

    def patch_files(self, graph: nx.DiGraph, changed_rel_paths: list[str]) -> nx.DiGraph:
        """Return *graph* with nodes for *changed_rel_paths* replaced by a fresh
        parse. Cross-file edges to unchanged nodes are preserved.

        When Neo4j is active, files whose SHA-256 hash hasn't changed since
        the last save are skipped — the diff listed them but their content
        is identical (e.g. only whitespace changed in another hunk).
        """
        actually_changed = self._filter_unchanged(changed_rel_paths)

        if actually_changed and self._neo4j:
            self._neo4j.delete_file_nodes(self._repo, actually_changed)

        changed_set = set(actually_changed)
        G = graph.copy()
        stale = [
            n for n, d in G.nodes(data=True)
            if d.get("file") in changed_set or n in changed_set
        ]
        G.remove_nodes_from(stale)

        parser = RepoParser(self.repo_path)
        parser.G = G
        for node_id, data in G.nodes(data=True):
            ntype = data.get("type", "")
            if ntype == "file":
                parser._file_nodes.add(node_id)
            elif ntype == "function":
                parser._name_index[data.get("name", "")].add(node_id)
                parent = data.get("parent_class")
                if parent:
                    parser._methods_by_class[(data["file"], parent)].add(node_id)

        for rel in actually_changed:
            abs_path = os.path.join(self.repo_path, rel)
            if not Path(abs_path).exists():
                continue
            ext = os.path.splitext(rel)[1]
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            parser._parse_file(abs_path, ext)

        parser._extract_call_edges()

        new_nodes = {n for n, d in G.nodes(data=True) if d.get("file") in changed_set}
        classifier = NodeClassifier(
            G,
            self.repo_path,
            strategy=self.strategy,
            neo4j_client=self._neo4j,
            repo_name=self._repo,
        )
        # Git metadata first — _classify_risk uses bug_fix_freq to determine
        # whether a function is critical, so the order matters.
        classifier._add_git_metadata(new_nodes)
        classifier._classify_risk(new_nodes)

        if actually_changed and self._neo4j:
            mini = nx.DiGraph()
            for nid, ndata in G.nodes(data=True):
                if ndata.get("file") in changed_set:
                    mini.add_node(nid, **ndata)
            for s, t, edata in G.edges(data=True):
                if G.nodes[s].get("file") in changed_set or G.nodes[t].get("file") in changed_set:
                    mini.add_edge(s, t, **edata)
            if mini.number_of_nodes() > 0:
                self._neo4j.save_graph(self._repo, mini)

        return G

    def _filter_unchanged(self, rel_paths: list[str]) -> list[str]:
        """Drop files whose SHA-256 already matches Neo4j's stored hash."""
        if not self._neo4j:
            return rel_paths
        stored = self._neo4j.get_file_hashes(self._repo)
        result = []
        for rel in rel_paths:
            abs_path = os.path.join(self.repo_path, rel)
            if not Path(abs_path).exists():
                result.append(rel)
                continue
            h = hashlib.sha256(Path(abs_path).read_bytes()).hexdigest()
            if stored.get(rel) != h:
                result.append(rel)
        return result
