"""Seed Pinecone with historical FastAPI review comments.

For each merged PR (excluding the eval set) we fetch the review thread ground
truth and insert every comment into Pinecone labelled by whether the developer
acted on it:

    addressed=True  → "upvote"   (reviewer's concern was addressed in a later commit)
    addressed=False → "downvote" (comment was ignored / PR merged as-is)

Usage
-----
    python scripts/seed_pinecone.py \\
        --repo fastapi/fastapi \\
        --max-prs 200 \\
        --exclude-prs 1234,5678,9012  # comma-separated eval PR numbers
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ares.config import AresConfig
from ares.integrations import GitHubClient, PineconeClient


def _comment_id(repo: str, pr_number: int, text: str) -> str:
    """Stable, collision-resistant ID for deduplication in Pinecone."""
    raw = f"{repo}::{pr_number}::{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _fetch_pr_items(
    github: GitHubClient,
    repo: str,
    pr_number: int,
) -> list[dict]:
    """Return upsert-ready items for a single PR, or [] on error."""
    try:
        threads = github.get_review_ground_truth(repo, pr_number)
    except Exception as exc:
        _log(f"PR #{pr_number}: ground-truth fetch failed — {exc}")
        return []
    items = []
    for entry in threads:
        text = (entry.get("comment") or "").strip()
        if not text:
            continue
        label = "upvote" if entry.get("addressed") else "downvote"
        items.append(
            {
                "id": _comment_id(repo, pr_number, text),
                "text": text,
                "label": label,
            }
        )
    return items


def _log(message: str) -> None:
    print(f"[seed] {message}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-seed Pinecone with historical FastAPI PR review comments."
    )
    parser.add_argument("--repo", default="fastapi/fastapi", help="GitHub repo (owner/name).")
    parser.add_argument(
        "--max-prs",
        type=int,
        default=200,
        help="Maximum number of recent merged PRs to harvest.",
    )
    parser.add_argument(
        "--exclude-prs",
        default="",
        help="Comma-separated PR numbers to skip (eval set).",
    )
    parser.add_argument(
        "--min-review-threads",
        type=int,
        default=1,
        help="Skip PRs with fewer than this many review threads.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of PRs to fetch in parallel.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = AresConfig.from_env()

    if not cfg.github_token:
        print("ERROR: GITHUB_TOKEN is required.", file=sys.stderr)
        return 1
    if not cfg.pinecone_api_key:
        print("ERROR: PINECONE_API_KEY is required.", file=sys.stderr)
        return 1

    exclude: set[int] = set()
    for part in args.exclude_prs.split(","):
        part = part.strip()
        if part.isdigit():
            exclude.add(int(part))

    github = GitHubClient(cfg.github_token)
    pinecone = PineconeClient(
        api_key=cfg.pinecone_api_key,
        index_name=cfg.pinecone_index_name,
        namespace=cfg.pinecone_namespace,
    )

    _log(f"Fetching up to {args.max_prs} recent merged PRs from {args.repo}...")
    recent = github.list_recent_merged_prs(args.repo, limit=args.max_prs)

    candidates = [
        pr["number"]
        for pr in recent
        if pr["number"] not in exclude
        and pr.get("review_thread_count", 0) >= args.min_review_threads
    ]
    _log(
        f"Found {len(recent)} merged PRs; {len(candidates)} candidates after excluding "
        f"{len(exclude)} eval PRs and low-thread PRs."
    )

    total_upvote = 0
    total_downvote = 0
    total_prs_seeded = 0
    batch: list[dict] = []
    BATCH_SIZE = 100

    def _flush(batch: list[dict]) -> None:
        if batch:
            pinecone.upsert_feedback(batch)

    parallelism = max(1, min(args.parallelism, len(candidates) or 1))
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_pr = {
            executor.submit(_fetch_pr_items, github, args.repo, pr_number): pr_number
            for pr_number in candidates
        }
        done = 0
        for future in as_completed(future_to_pr):
            pr_number = future_to_pr[future]
            done += 1
            items = future.result()
            if not items:
                _log(f"PR #{pr_number}: 0 comments ({done}/{len(candidates)}).")
                continue

            upvotes = sum(1 for item in items if item["label"] == "upvote")
            downvotes = len(items) - upvotes
            total_upvote += upvotes
            total_downvote += downvotes
            total_prs_seeded += 1
            batch.extend(items)
            _log(
                f"PR #{pr_number}: {len(items)} comments "
                f"(+{upvotes} upvote / -{downvotes} downvote) "
                f"[{done}/{len(candidates)}]."
            )
            if len(batch) >= BATCH_SIZE:
                _flush(batch)
                batch = []

    _flush(batch)

    total = total_upvote + total_downvote
    _log(
        f"Done. Seeded {total} comments from {total_prs_seeded} PRs — "
        f"upvote={total_upvote} ({total_upvote/total:.1%}), "
        f"downvote={total_downvote} ({total_downvote/total:.1%})."
        if total else "Done. No comments seeded."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
