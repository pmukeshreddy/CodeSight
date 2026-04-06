from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

logger = logging.getLogger(__name__)

# Map networkx edge type strings → Neo4j relationship types
_EDGE_TYPE_MAP = {
    "calls": "CALLS",
    "contains": "CONTAINS",
    "tests": "TESTS",
    "co_changes": "CO_CHANGES",
    "imports": "IMPORTS",
}
# Map networkx node type strings → Neo4j node labels
_NODE_LABEL_MAP = {
    "file": "AresFile",
    "function": "AresFunction",
    "class": "AresClass",
}


def _node_label(node_type: str) -> str:
    return _NODE_LABEL_MAP.get(node_type, "AresNode")


def _rel_type(edge_type: str) -> str:
    return _EDGE_TYPE_MAP.get(edge_type, "RELATED")


def _serialize_props(data: dict) -> dict:
    result: dict[str, Any] = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
        elif isinstance(v, list):
            if not v:
                continue
            if all(isinstance(i, str) for i in v):
                result[k] = v
            elif all(isinstance(i, (int, float)) for i in v):
                result[k] = v
            else:
                result[k] = json.dumps(v)
    return result


class Neo4jClient:
    """
    Persistent, graph-native store for ARES code graphs.

    Schema:
      Nodes  — labels :AresFile | :AresFunction | :AresClass, all tagged :AresNode
               Properties: node_id (repo::local_id), repo, local_id, + all parsed fields
               AresFile also stores content_hash for incremental re-indexing.

      Edges  — real relationship types: :CALLS :CONTAINS :TESTS :CO_CHANGES :IMPORTS
               Properties: types (list), count (co_changes)

    Traversal methods use Cypher graph patterns directly — callers, callees,
    transitive counts — so BFS runs inside Neo4j, not in Python.
    """

    def __init__(self, uri: str = "", user: str = "neo4j", password: str = ""):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = self._connect() if uri else None

    @property
    def available(self) -> bool:
        return self._driver is not None

    # ------------------------------------------------------------------
    # Connection + schema
    # ------------------------------------------------------------------

    def _connect(self):
        if GraphDatabase is None:
            raise RuntimeError("neo4j package not installed; pip install neo4j>=5.0")
        try:
            from neo4j import NotificationDisabledCategory
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                notifications_disabled_categories=[NotificationDisabledCategory.UNRECOGNIZED],
            )
        except (ImportError, TypeError):
            # Older neo4j driver without notification filtering
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        driver.verify_connectivity()
        self._ensure_schema(driver)
        return driver

    def _ensure_schema(self, driver) -> None:
        with driver.session() as s:
            # Uniqueness constraint on node_id — covers all node types
            s.run(
                "CREATE CONSTRAINT ares_node_id IF NOT EXISTS "
                "FOR (n:AresNode) REQUIRE n.node_id IS UNIQUE"
            )
            # Index for fast repo+file lookups used in incremental patching
            s.run(
                "CREATE INDEX ares_repo_file IF NOT EXISTS "
                "FOR (n:AresNode) ON (n.repo, n.file)"
            )

    # ------------------------------------------------------------------
    # Persistence: save / load
    # ------------------------------------------------------------------

    def save_graph(self, repo: str, G) -> None:
        """Write networkx DiGraph to Neo4j with batched UNWIND queries."""
        import time

        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        BATCH = 500

        # --- nodes by label ---
        nodes_by_label: dict[str, list[dict]] = {}
        for local_id, data in G.nodes(data=True):
            node_type = data.get("type", "")
            label = _node_label(node_type)
            props = _serialize_props(data)
            props["node_id"] = f"{repo}::{local_id}"
            props["local_id"] = local_id
            props["repo"] = repo
            if node_type == "file":
                src = data.get("source", "")
                props["content_hash"] = hashlib.sha256(
                    src.encode("utf-8", errors="ignore")
                ).hexdigest()
            nodes_by_label.setdefault(label, []).append(props)

        t0 = time.perf_counter()
        done = 0
        with self._driver.session() as s:
            for label, batch_nodes in nodes_by_label.items():
                for i in range(0, len(batch_nodes), BATCH):
                    chunk = batch_nodes[i : i + BATCH]
                    result = s.run(
                        f"UNWIND $rows AS row "
                        f"MERGE (n:AresNode {{node_id: row.node_id}}) "
                        f"SET n:{label}, n += row",
                        rows=chunk,
                    )
                    summary = result.consume()
                    done += len(chunk)
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_nodes - done) / rate if rate > 0 else 0
                    print(f"\r  Nodes: {done}/{total_nodes} ({rate:.0f}/s, ETA {eta:.0f}s)", end="", flush=True)
        print()

        # --- edges by relationship type ---
        edges_by_rel: dict[str, list[dict]] = {}
        for src_id, dst_id, data in G.edges(data=True):
            raw_types = set(data.get("types") or [])
            if data.get("type"):
                raw_types.add(data["type"])
            for edge_type in raw_types:
                rel = _rel_type(edge_type)
                row: dict[str, Any] = {
                    "src": f"{repo}::{src_id}",
                    "dst": f"{repo}::{dst_id}",
                }
                if data.get("count"):
                    row["count"] = int(data["count"])
                if raw_types - {edge_type}:
                    row["also"] = sorted(raw_types - {edge_type})
                edges_by_rel.setdefault(rel, []).append(row)

        t1 = time.perf_counter()
        done = 0
        total_rels = sum(len(v) for v in edges_by_rel.values())
        with self._driver.session() as s:
            for rel, batch_edges in edges_by_rel.items():
                for i in range(0, len(batch_edges), BATCH):
                    chunk = batch_edges[i : i + BATCH]
                    result = s.run(
                        f"UNWIND $rows AS row "
                        f"MATCH (a:AresNode {{node_id: row.src}}), (b:AresNode {{node_id: row.dst}}) "
                        f"MERGE (a)-[r:{rel}]->(b) "
                        f"SET r += row",
                        rows=chunk,
                    )
                    summary = result.consume()
                    done += len(chunk)
                    elapsed = time.perf_counter() - t1
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_rels - done) / rate if rate > 0 else 0
                    print(f"\r  Edges: {done}/{total_rels} ({rate:.0f}/s, ETA {eta:.0f}s)", end="", flush=True)
        print()

        # Verify data was persisted
        with self._driver.session() as s:
            r = s.run(
                "MATCH (n:AresNode {repo: $repo}) RETURN count(n) AS nodes",
                repo=repo,
            ).single()
            node_count = r["nodes"] if r else 0
            r2 = s.run(
                "MATCH (n:AresNode {repo: $repo})-[r]->() RETURN count(r) AS rels",
                repo=repo,
            ).single()
            rel_count = r2["rels"] if r2 else 0
            print(f"  Verified in Neo4j: {node_count} nodes, {rel_count} relationships", flush=True)

    def load_graph(self, repo: str):
        """Reconstruct nx.DiGraph from Neo4j. Returns None if repo has no data."""
        try:
            import networkx as nx
        except ImportError:
            return None

        G = nx.DiGraph()
        with self._driver.session() as s:
            # nodes
            for record in s.run(
                "MATCH (n:AresNode {repo: $repo}) "
                "RETURN n.local_id AS lid, properties(n) AS props, labels(n) AS lbls",
                repo=repo,
            ):
                lid = record["lid"]
                props = {
                    k: v for k, v in record["props"].items()
                    if k not in {"node_id", "repo", "local_id", "content_hash"}
                }
                # Deserialise JSON-encoded list-of-dicts fields
                for key in ("calls",):
                    if isinstance(props.get(key), str):
                        try:
                            props[key] = json.loads(props[key])
                        except (ValueError, TypeError):
                            pass
                G.add_node(lid, **props)

            # edges — one query per relationship type so we get proper type back
            for rel_type, ares_type in [
                ("CALLS", "calls"), ("CONTAINS", "contains"), ("TESTS", "tests"),
                ("CO_CHANGES", "co_changes"), ("IMPORTS", "imports"), ("RELATED", "related"),
            ]:
                for record in s.run(
                    f"MATCH (s:AresNode {{repo: $repo}})-[r:{rel_type}]->(d:AresNode {{repo: $repo}}) "
                    f"RETURN s.local_id AS s, d.local_id AS d, r.count AS cnt, r.also AS also",
                    repo=repo,
                ):
                    sv, dv = record["s"], record["d"]
                    if sv is None or dv is None:
                        continue
                    if G.has_edge(sv, dv):
                        existing = G.edges[sv, dv]
                        types = set(existing.get("types", []))
                        types.add(ares_type)
                        existing["types"] = sorted(types)
                    else:
                        props: dict[str, Any] = {"type": ares_type, "types": [ares_type]}
                        if record["cnt"]:
                            props["count"] = int(record["cnt"])
                        G.add_edge(sv, dv, **props)

        return G if G.number_of_nodes() > 0 else None

    # ------------------------------------------------------------------
    # Incremental patching helpers
    # ------------------------------------------------------------------

    def has_graph(self, repo: str) -> bool:
        with self._driver.session() as s:
            r = s.run(
                "MATCH (n:AresNode {repo: $repo}) RETURN count(n) AS cnt LIMIT 1",
                repo=repo,
            ).single()
            return bool(r and r["cnt"] > 0)

    def get_file_hashes(self, repo: str) -> dict[str, str]:
        """Return {local_file_path: sha256_hex} for all AresFile nodes."""
        with self._driver.session() as s:
            return {
                r["fid"]: r["h"]
                for r in s.run(
                    "MATCH (n:AresFile {repo: $repo}) RETURN n.local_id AS fid, n.content_hash AS h",
                    repo=repo,
                )
                if r["h"]
            }

    def delete_file_nodes(self, repo: str, file_paths: list[str]) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (n:AresNode {repo: $repo}) WHERE n.file IN $files DETACH DELETE n",
                repo=repo,
                files=file_paths,
            )

    def update_node_metadata(self, repo: str, local_id: str, metadata: dict) -> None:
        props = _serialize_props(metadata)
        if not props:
            return
        with self._driver.session() as s:
            s.run(
                "MATCH (n:AresNode {node_id: $nid}) SET n += $props",
                nid=f"{repo}::{local_id}",
                props=props,
            )

    # ------------------------------------------------------------------
    # Graph traversal — Cypher runs the BFS, Python gets results
    # ------------------------------------------------------------------

    def query_callers(self, repo: str, node_id: str, max_hops: int = 3) -> list[dict]:
        """
        BFS up: all AresFunction nodes that transitively call node_id.
        Uses variable-length CALLS path — runs entirely inside Neo4j.
        Returns list of {local_id, hop, ...node_data}.
        """
        # Neo4j Aura (Cypher 25) does not support parameterized variable-length
        # ranges ($hops), so we interpolate the int directly. max_hops is always
        # a small trusted integer from internal code.
        query = (
            f"MATCH path = (caller:AresFunction {{repo: $repo}})"
            f"-[:CALLS*1..{int(max_hops)}]->"
            f"(f:AresNode {{node_id: $nid}}) "
            f"RETURN DISTINCT caller, length(path) AS hop "
            f"ORDER BY hop"
        )
        with self._driver.session() as s:
            results = []
            for record in s.run(query, repo=repo, nid=f"{repo}::{node_id}"):
                node = dict(record["caller"])
                node["hop"] = record["hop"]
                node["local_id"] = node.get("local_id", "")
                results.append(node)
            return results

    def query_callees(self, repo: str, node_id: str, max_hops: int = 3) -> list[dict]:
        """BFS down: all AresFunction nodes transitively called by node_id."""
        query = (
            f"MATCH path = (f:AresNode {{node_id: $nid}})"
            f"-[:CALLS*1..{int(max_hops)}]->"
            f"(callee:AresFunction {{repo: $repo}}) "
            f"RETURN DISTINCT callee, length(path) AS hop "
            f"ORDER BY hop"
        )
        with self._driver.session() as s:
            results = []
            for record in s.run(query, repo=repo, nid=f"{repo}::{node_id}"):
                node = dict(record["callee"])
                node["hop"] = record["hop"]
                node["local_id"] = node.get("local_id", "")
                results.append(node)
            return results

    def query_transitive_caller_count(self, repo: str, node_id: str) -> int:
        """Count all functions that transitively reach node_id via CALLS edges."""
        with self._driver.session() as s:
            r = s.run(
                "MATCH (caller:AresFunction {repo: $repo})-[:CALLS*1..]->(f:AresNode {node_id: $nid}) "
                "RETURN count(DISTINCT caller) AS cnt",
                repo=repo,
                nid=f"{repo}::{node_id}",
            ).single()
            return int(r["cnt"]) if r else 0

    def query_direct_caller_count(self, repo: str, node_id: str) -> int:
        """Count functions with a direct CALLS edge to node_id."""
        with self._driver.session() as s:
            r = s.run(
                "MATCH (caller:AresFunction {repo: $repo})-[:CALLS]->(f:AresNode {node_id: $nid}) "
                "RETURN count(caller) AS cnt",
                repo=repo,
                nid=f"{repo}::{node_id}",
            ).single()
            return int(r["cnt"]) if r else 0

    def query_tests(self, repo: str, node_id: str) -> list[dict]:
        """Return all test functions that directly test node_id."""
        with self._driver.session() as s:
            return [
                {**dict(r["t"]), "local_id": r["t"].get("local_id", "")}
                for r in s.run(
                    "MATCH (t:AresFunction {repo: $repo})-[:TESTS]->(f:AresNode {node_id: $nid}) "
                    "RETURN t",
                    repo=repo,
                    nid=f"{repo}::{node_id}",
                )
            ]

    def query_co_changes(self, repo: str, node_id: str) -> list[dict]:
        """Return nodes that co-change with node_id, ordered by co-change count."""
        with self._driver.session() as s:
            return [
                {
                    "local_id": r["n"].get("local_id", ""),
                    "file": r["n"].get("file", ""),
                    "count": r["cnt"] or 0,
                }
                for r in s.run(
                    "MATCH (f:AresNode {node_id: $nid})-[r:CO_CHANGES]->(n:AresNode {repo: $repo}) "
                    "RETURN n, r.count AS cnt "
                    "ORDER BY cnt DESC",
                    repo=repo,
                    nid=f"{repo}::{node_id}",
                )
            ]

    def query_node(self, repo: str, local_id: str) -> dict | None:
        """Fetch a single node's properties."""
        with self._driver.session() as s:
            r = s.run(
                "MATCH (n:AresNode {node_id: $nid}) RETURN properties(n) AS props",
                nid=f"{repo}::{local_id}",
            ).single()
            if not r:
                return None
            props = dict(r["props"])
            props.pop("node_id", None)
            props.pop("repo", None)
            return props

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
