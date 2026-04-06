"""Build the Neo4j knowledge graph for a GitHub repository.

Clones the repository at the specified branch (default: main), parses the
codebase with tree-sitter, extracts the call graph, classifies node risk,
adds git metadata, and stores everything in Neo4j.

Once the graph exists, ``pipeline.review_pr()`` will incrementally patch
only the files changed in each PR instead of re-indexing from scratch.

Required env vars
-----------------
    GITHUB_TOKEN          GitHub personal access token
    ARES_NEO4J_URI        e.g. neo4j://localhost:7687
    ARES_NEO4J_PASSWORD   Neo4j password

Usage
-----
    python scripts/build_graph.py \\
        --repo fastapi/fastapi \\
        --branch main \\
        --clone-depth 100
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ares.config import AresConfig
from ares.pipeline import Pipeline


def _log(message: str) -> None:
    print(f"[build_graph] {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Neo4j knowledge graph for a GitHub repository."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repo in owner/name form (e.g. fastapi/fastapi).",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to clone and index (default: main).",
    )
    parser.add_argument(
        "--clone-dir",
        default="",
        help="Directory to clone into (default: .ares-workdir/<repo>/base).",
    )
    parser.add_argument(
        "--clone-depth",
        type=int,
        default=100,
        help="Git clone depth — deeper means richer git metadata (default: 100).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if a graph already exists in Neo4j for this repo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = build_parser().parse_args(argv)
    cfg = AresConfig.from_env()

    if not cfg.github_token:
        print("ERROR: GITHUB_TOKEN is required.", file=sys.stderr)
        return 1
    if not cfg.neo4j_uri:
        print("ERROR: NEO4J_URI is required.", file=sys.stderr)
        return 1
    if not cfg.neo4j_password:
        print("ERROR: NEO4J_PASSWORD is required.", file=sys.stderr)
        return 1

    pipeline = Pipeline(cfg)
    # Neo4j connection is validated during Pipeline construction;
    # if we reach here, it's connected.

    # Check if graph already exists
    if not args.force and pipeline.neo4j.has_graph(args.repo):
        _log(f"Graph for {args.repo} already exists in Neo4j. Use --force to rebuild.")
        return 0

    # Determine clone directory
    clone_dir = args.clone_dir or str(
        Path(cfg.workspace_root) / args.repo.replace("/", "__") / "base"
    )

    # Clone the repository
    _log(f"Cloning {args.repo}@{args.branch} into {clone_dir} (depth={args.clone_depth})...")
    t0 = time.perf_counter()
    pipeline.github.clone_repo(args.repo, args.branch, clone_dir, depth=args.clone_depth)
    _log(f"Clone complete in {time.perf_counter() - t0:.1f}s.")

    # Build and store the graph
    _log("Indexing repository into Neo4j...")
    pipeline._active_repo_name = args.repo
    t0 = time.perf_counter()
    pipeline.index_repo(clone_dir, repo_name=args.repo)
    elapsed = time.perf_counter() - t0

    node_count = pipeline.graph.number_of_nodes() if pipeline.graph else 0
    edge_count = pipeline.graph.number_of_edges() if pipeline.graph else 0
    _log(
        f"Done. Graph stored in Neo4j: {node_count} nodes, {edge_count} edges "
        f"({elapsed:.1f}s)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
