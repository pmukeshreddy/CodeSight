from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ares.config import AresConfig
from ares.evaluate import Evaluator
from ares.pipeline import Pipeline
from ares.review_scope import is_maintenance_only_pr, reviewable_source_files

HISTORICAL_EVAL_COMMENT_CAP = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample generated review comments across PRs until a target comment count is reached."
    )
    parser.add_argument("--repo", default="fastapi/fastapi", help="GitHub repo in owner/name form.")
    parser.add_argument(
        "--target-comments",
        type=int,
        default=100,
        help="Stop once this many generated comments have been accumulated.",
    )
    parser.add_argument(
        "--max-inspected-prs",
        type=int,
        default=300,
        help="Inspect at most this many recent merged PRs when building the PR pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to shuffle candidate PRs.",
    )
    parser.add_argument(
        "--workspace-root",
        default="/tmp/ares-workdir",
        help="Workspace root for pipeline state.",
    )
    parser.add_argument(
        "--clone-root",
        default="/tmp/ares-comment-sample",
        help="Directory under which each sampled PR gets its own clone.",
    )
    parser.add_argument(
        "--prs",
        default="",
        help="Optional comma-separated PR numbers. If provided, random sampling is skipped.",
    )
    parser.add_argument(
        "--min-human-comments",
        type=int,
        default=1,
        help="Require at least this many human review comments when building the PR pool.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of PRs to evaluate concurrently in each batch.",
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Branch to use for the shared base graph (e.g. main, master).",
    )
    parser.add_argument(
        "--snapshot-mode",
        default="final",
        choices=["final", "first-human-review"],
        help="Review the latest PR state or the first human-reviewed snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = AresConfig.from_env(workspace_root=args.workspace_root)
    eval_max_comments = effective_eval_max_comments(cfg.max_comments, int(args.target_comments))
    cfg.max_comments = eval_max_comments
    pipeline = Pipeline(cfg)
    if pipeline.github is None:
        raise RuntimeError("GITHUB_TOKEN is required.")
    if not cfg.llm_api_key:
        raise RuntimeError("An LLM API key is required to generate comments.")

    # Evaluation should not post anything back to GitHub or mutate feedback state.
    pipeline.github.post_review_comments = lambda *args, **kwargs: None
    pipeline.collector.record_posted_comments = lambda *args, **kwargs: []

    progress(
        f"Sampling generated comments from {args.repo} until {max(1, int(args.target_comments))} comments are collected."
    )
    progress(
        "Using {count} comments per PR for this eval run.".format(
            count=eval_max_comments,
        )
    )
    if eval_max_comments > int(os.getenv("ARES_MAX_COMMENTS", "3")):
        progress(
            "Auto-raised eval comment cap to {count} for historical sampling. Set ARES_MAX_COMMENTS explicitly to override.".format(
                count=eval_max_comments,
            )
        )
    (
        pr_numbers,
        inspected_prs,
        prefetched_ground_truth,
        candidate_review_counts,
        preflight_rejected_counts,
    ) = candidate_pr_numbers(
        pipeline,
        repo_name=args.repo,
        max_inspected_prs=args.max_inspected_prs,
        prs_arg=args.prs,
        seed=args.seed,
        min_human_comments=max(0, int(args.min_human_comments)),
        preflight_parallelism=max(1, int(args.parallelism)),
    )
    progress(
        "Prepared {approved} candidate PRs after inspecting {inspected} and prefiltering out "
        "{rejected} PRs (no_reviewable_source_files={no_source}, maintenance_only_pr={maintenance}).".format(
            approved=len(pr_numbers),
            inspected=inspected_prs,
            rejected=sum(preflight_rejected_counts.values()),
            no_source=preflight_rejected_counts.get("no_reviewable_source_files", 0),
            maintenance=preflight_rejected_counts.get("maintenance_only_pr", 0),
        )
    )
    # Ensure the base graph exists in Neo4j by indexing the base branch.
    progress("Checking base graph in Neo4j...")
    _ensure_base_graph(pipeline, repo_name=args.repo, base_branch=args.base_branch, base_clone_root=Path(args.clone_root))
    result = sample_comment_metrics(
        config=cfg,
        repo_name=args.repo,
        pr_numbers=pr_numbers,
        target_comments=max(1, int(args.target_comments)),
        clone_root=Path(args.clone_root),
        prefetched_ground_truth=prefetched_ground_truth,
        candidate_review_counts=candidate_review_counts,
        max_comments_per_pr=eval_max_comments,
        parallelism=max(1, int(args.parallelism)),
        snapshot_mode=args.snapshot_mode,
    )
    result["repo_name"] = args.repo
    result["target_comments"] = max(1, int(args.target_comments))
    result["seed"] = args.seed
    result["inspected_prs"] = inspected_prs
    result["preflight_rejected_counts"] = preflight_rejected_counts
    result["max_comments_per_pr"] = eval_max_comments
    progress(
        "Done. {n} comments across {prs} PRs — address_rate={ar:.1%}  precision={prec:.1%}  plausible={pl:.1%}".format(
            n=result.get("sampled_comments", 0),
            prs=result.get("sampled_pr_count", 0),
            ar=result.get("address_rate", 0.0),
            prec=result.get("precision", 0.0),
            pl=result.get("plausible_rate", 0.0),
        )
    )
    print(json.dumps(result, indent=2))
    return 0


def effective_eval_max_comments(current_max_comments: int, target_comments: int) -> int:
    if target_comments >= 100 and current_max_comments < HISTORICAL_EVAL_COMMENT_CAP:
        return HISTORICAL_EVAL_COMMENT_CAP
    return current_max_comments


def _ensure_base_graph(
    pipeline: Pipeline,
    repo_name: str,
    base_branch: str,
    base_clone_root: Path,
) -> None:
    """Index the base branch into Neo4j if no graph exists yet."""
    if pipeline.neo4j.available and pipeline.neo4j.has_graph(repo_name):
        progress(f"Reusing existing graph in Neo4j for {repo_name}.")
        return
    base_dir = base_clone_root / "base"
    progress(f"Cloning {repo_name}@{base_branch} to build base graph in Neo4j...")
    try:
        pipeline.github.clone_repo(repo_name, base_branch, str(base_dir), depth=100)
        pipeline.index_repo(str(base_dir), repo_name=repo_name)
        progress("Base graph stored in Neo4j.")
    except Exception as exc:
        progress(f"Base graph build failed ({exc}); each PR will be fully indexed.")


def candidate_pr_numbers(
    pipeline: Pipeline,
    repo_name: str,
    max_inspected_prs: int,
    prs_arg: str,
    seed: int,
    min_human_comments: int,
    preflight_parallelism: int,
) -> tuple[list[int], int, dict[int, list[dict]], dict[int, int], dict[str, int]]:
    if prs_arg.strip():
        prs = [int(part.strip()) for part in prs_arg.split(",") if part.strip()]
        progress(f"Using explicit PR list with {len(prs)} entries.")
        approved, rejected_counts = preflight_candidate_prs(
            pipeline,
            repo_name,
            prs,
            {pr_number: 0 for pr_number in prs},
            parallelism=preflight_parallelism,
        )
        prefetched = {
            pr_number: pipeline.github.get_review_ground_truth(repo_name, pr_number)
            for pr_number in approved
        }
        counts = {pr_number: len(prefetched.get(pr_number, [])) for pr_number in approved}
        return approved, len(prs), prefetched, counts, rejected_counts
    merged_prs: list[int] = []
    prefetched_ground_truth: dict[int, list[dict]] = {}
    candidate_review_counts: dict[int, int] = {}
    progress(f"Fetching up to {max_inspected_prs} recent merged PRs from {repo_name}...")
    recent_prs = pipeline.github.list_recent_merged_prs(repo_name, max_inspected_prs)
    if recent_prs:
        for inspected, pr in enumerate(recent_prs, start=1):
            if inspected % 25 == 0:
                progress(f"Scanned {inspected} PRs, found {len(merged_prs)} candidate PRs.")
            pr_number = int(pr.get("number", 0) or 0)
            review_count = int(pr.get("review_thread_count", 0) or 0)
            if pr_number <= 0 or review_count < min_human_comments:
                continue
            merged_prs.append(pr_number)
            candidate_review_counts[pr_number] = review_count
        random.Random(seed).shuffle(merged_prs)
        approved_prs, rejected_counts = preflight_candidate_prs(
            pipeline,
            repo_name,
            merged_prs,
            candidate_review_counts,
            parallelism=preflight_parallelism,
        )
        approved_counts = {pr_number: candidate_review_counts[pr_number] for pr_number in approved_prs}
        progress(f"Fetching ground truth for {len(approved_prs)} approved PRs (parallel)...")
        done = 0
        def _fetch_gt(pr_num: int) -> tuple[int, list[dict]]:
            return pr_num, pipeline.github.get_review_ground_truth(repo_name, pr_num)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for pr_num, gt in pool.map(_fetch_gt, approved_prs):
                prefetched_ground_truth[pr_num] = gt
                approved_counts[pr_num] = len(gt)
                done += 1
                if done == 1 or done % 20 == 0 or done == len(approved_prs):
                    progress(f"  Ground truth: {done}/{len(approved_prs)}")
        return approved_prs, len(recent_prs), prefetched_ground_truth, approved_counts, rejected_counts
    repo = pipeline.github.gh.get_repo(repo_name)
    rejected_prs: set[int] = set()
    inspected = 0
    progress("GraphQL PR listing unavailable, falling back to review-comment scan.")
    for comment in repo.get_pulls_comments(sort="updated", direction="desc"):
        inspected += 1
        if inspected % 25 == 0:
            progress(f"Scanned {inspected} review comments, found {len(merged_prs)} candidate PRs.")
        if inspected > max_inspected_prs:
            break
        pr_number = parse_pr_number(getattr(comment, "pull_request_url", ""))
        if pr_number is None or pr_number in candidate_review_counts or pr_number in rejected_prs:
            continue
        try:
            pr = repo.get_pull(pr_number)
        except Exception:
            rejected_prs.add(pr_number)
            continue
        if getattr(pr, "merged_at", None) is None:
            rejected_prs.add(pr_number)
            continue
        ground_truth = pipeline.github.get_review_ground_truth(repo_name, pr_number)
        if len(ground_truth) < min_human_comments:
            rejected_prs.add(pr_number)
            continue
        prefetched_ground_truth[pr_number] = ground_truth
        candidate_review_counts[pr_number] = len(ground_truth)
        merged_prs.append(pr_number)
    random.Random(seed).shuffle(merged_prs)
    approved_prs, rejected_counts = preflight_candidate_prs(
        pipeline,
        repo_name,
        merged_prs,
        candidate_review_counts,
        parallelism=preflight_parallelism,
    )
    approved_counts = {pr_number: candidate_review_counts[pr_number] for pr_number in approved_prs}
    approved_ground_truth = {pr_number: prefetched_ground_truth[pr_number] for pr_number in approved_prs if pr_number in prefetched_ground_truth}
    return approved_prs, inspected, approved_ground_truth, approved_counts, rejected_counts


def preflight_candidate_prs(
    pipeline: Pipeline,
    repo_name: str,
    pr_numbers: list[int],
    candidate_review_counts: dict[int, int],
    parallelism: int,
) -> tuple[list[int], dict[str, int]]:
    if not pr_numbers:
        return [], {"no_reviewable_source_files": 0, "maintenance_only_pr": 0}
    progress(f"Preflighting {len(pr_numbers)} candidate PRs for reviewable source changes...")
    approved: list[int] = []
    rejected_counts = {"no_reviewable_source_files": 0, "maintenance_only_pr": 0}
    with ThreadPoolExecutor(max_workers=min(max(1, parallelism), len(pr_numbers))) as executor:
        future_to_pr = {
            executor.submit(preflight_pr_candidate, pipeline, repo_name, pr_number): pr_number
            for pr_number in pr_numbers
        }
        completed = 0
        for future in as_completed(future_to_pr):
            completed += 1
            pr_number = future_to_pr[future]
            try:
                result = future.result()
            except Exception:
                result = {"status": "rejected", "reason": "no_reviewable_source_files"}
            if result.get("status") == "approved":
                approved.append(pr_number)
            else:
                reason = result.get("reason", "no_reviewable_source_files")
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            if completed % 25 == 0 or completed == len(pr_numbers):
                progress(
                    "Preflighted {done}/{total} PRs, approved {approved}, rejected {rejected} "
                    "(no_reviewable_source_files={no_source}, maintenance_only_pr={maintenance}).".format(
                        done=completed,
                        total=len(pr_numbers),
                        approved=len(approved),
                        rejected=sum(rejected_counts.values()),
                        no_source=rejected_counts.get("no_reviewable_source_files", 0),
                        maintenance=rejected_counts.get("maintenance_only_pr", 0),
                    )
                )
    approved.sort(key=lambda pr_number: pr_numbers.index(pr_number))
    return approved, rejected_counts


def preflight_pr_candidate(
    pipeline: Pipeline,
    repo_name: str,
    pr_number: int,
) -> dict:
    overview = pipeline.github.get_pr_overview(repo_name, pr_number)
    changed_files = overview.get("changed_files", [])
    reviewable_files = reviewable_source_files(changed_files)
    text = "\n".join([overview.get("title", ""), overview.get("description", "")]).strip()
    if is_maintenance_only_pr(text, changed_files):
        return {"status": "rejected", "reason": "maintenance_only_pr"}
    if not reviewable_files:
        return {"status": "rejected", "reason": "no_reviewable_source_files"}
    return {
        "status": "approved",
        "reviewable_changed_files": reviewable_files,
    }


def sample_comment_metrics(
    config: AresConfig,
    repo_name: str,
    pr_numbers: list[int],
    target_comments: int,
    clone_root: Path,
    prefetched_ground_truth: dict[int, list[dict]] | None = None,
    candidate_review_counts: dict[int, int] | None = None,
    max_comments_per_pr: int | None = None,
    parallelism: int = 1,
    snapshot_mode: str = "final",
) -> dict:
    clone_root.mkdir(parents=True, exist_ok=True)
    prefetched_ground_truth = prefetched_ground_truth or {}
    candidate_review_counts = candidate_review_counts or {}
    total_generated = 0
    total_matched = 0
    total_addressed = 0
    total_plausible = 0
    total_verified = 0
    used_prs: list[dict] = []
    skipped_prs: list[dict] = []

    # Reuse a single Pipeline instance — avoids reconnecting Neo4j/Pinecone
    # and reloading the embedding model for every PR.
    shared_pipeline = Pipeline(config)
    shared_pipeline.github.post_review_comments = lambda *args, **kwargs: None
    shared_pipeline.collector.record_posted_comments = lambda *args, **kwargs: []

    for idx, pr_number in enumerate(pr_numbers):
        if total_generated >= target_comments:
            break
        progress(
            "PR #{pr} [{idx}/{total_prs}]: starting ({collected}/{target} comments so far).".format(
                pr=pr_number,
                idx=idx + 1,
                total_prs=len(pr_numbers),
                collected=total_generated,
                target=target_comments,
            )
        )
        try:
            result = review_pr_sample(
                config.workspace_root,
                repo_name,
                pr_number,
                str(clone_root / f"pr-{pr_number}"),
                prefetched_ground_truth.get(pr_number, []),
                max_comments_per_pr or config.max_comments,
                pipeline=shared_pipeline,
                snapshot_mode=snapshot_mode,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            progress(f"PR #{pr_number}: failed with {exc}.")
            skipped_prs.append({"pr_number": pr_number, "reason": f"error: {exc}"})
            continue

        if result.get("status") != "ok":
            reason = result.get("reason", "unknown")
            if reason == "no_human_review_comments":
                progress(f"PR #{pr_number}: skipped, no human review comments.")
            elif reason in {
                "no_generated_comments",
                "no_targets",
                "no_indexed_functions_in_reviewable_files",
                "review_loop_zero_survivors",
                "ranker_zero_comments",
                "bundle_dropped_all",
            }:
                progress(f"PR #{pr_number}: generated 0 comments, skipping ({reason}).")
            else:
                progress(f"PR #{pr_number}: failed with {reason}.")
            skipped_prs.append({"pr_number": pr_number, "reason": reason})
            continue

        remaining = target_comments - total_generated
        if remaining <= 0:
            break

        selected = result.get("comment_outcomes", [])[:remaining]
        matched = sum(1 for item in selected if item.get("matched"))
        addressed = sum(1 for item in selected if item.get("addressed"))
        plausible = sum(1 for item in selected if item.get("plausible"))
        verified = sum(1 for item in selected if item.get("verified"))

        total_generated += len(selected)
        total_matched += matched
        total_addressed += addressed
        total_plausible += plausible
        total_verified += verified
        running_address_rate = total_addressed / total_generated if total_generated else 0.0
        running_precision = total_matched / total_generated if total_generated else 0.0
        running_plausible = total_plausible / total_generated if total_generated else 0.0
        progress(
            "PR #{pr}: kept {kept} comments, matched {matched}, plausible {plausible}, addressed {addressed}, "
            "total {total}/{target} in {seconds:.1f}s. "
            "[running: address_rate={ar:.1%} precision={prec:.1%} plausible={pl:.1%}]".format(
                pr=pr_number,
                kept=len(selected),
                matched=matched,
                plausible=plausible,
                addressed=addressed,
                total=total_generated,
                target=target_comments,
                seconds=result.get("elapsed_seconds", 0.0),
                ar=running_address_rate,
                prec=running_precision,
                pl=running_plausible,
            )
        )
        used_prs.append(
            {
                "pr_number": pr_number,
                "comments_generated": len(selected),
                "matched": matched,
                "addressed": addressed,
                "plausible": plausible,
                "verified": verified,
                "address_rate": round(addressed / len(selected), 3) if selected else 0.0,
                "precision": round(matched / len(selected), 3) if selected else 0.0,
                "plausible_rate": round(plausible / len(selected), 3) if selected else 0.0,
            }
        )

    return {
        "sampled_comments": total_generated,
        "sampled_pr_count": len(used_prs),
        "addressed_comments": total_addressed,
        "matched_comments": total_matched,
        "plausible_comments": total_plausible,
        "verified_comments": total_verified,
        "address_rate": round(total_addressed / total_generated, 3) if total_generated else 0.0,
        "precision": round(total_matched / total_generated, 3) if total_generated else 0.0,
        "plausible_rate": round(total_plausible / total_generated, 3) if total_generated else 0.0,
        "verified_rate": round(total_verified / total_generated, 3) if total_generated else 0.0,
        "used_prs": used_prs,
        "skipped_prs": skipped_prs,
    }


def progress(message: str) -> None:
    print(f"[progress] {message}", file=sys.stderr, flush=True)


def parse_pr_number(pull_request_url: str) -> int | None:
    match = re.search(r"/pulls/(\d+)$", pull_request_url or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def review_pr_sample(
    workspace_root: str,
    repo_name: str,
    pr_number: int,
    target_dir: str,
    human_comments: list[dict],
    max_comments_per_pr: int,
    pipeline: Pipeline | None = None,
    snapshot_mode: str = "final",
) -> dict:
    start = time.time()
    if pipeline is None:
        cfg = AresConfig.from_env(workspace_root=workspace_root)
        cfg.max_comments = max(1, int(max_comments_per_pr))
        pipeline = Pipeline(cfg)
        pipeline.github.post_review_comments = lambda *args, **kwargs: None
        pipeline.collector.record_posted_comments = lambda *args, **kwargs: []
    if pipeline.github is None:
        return {"status": "failed", "reason": "missing_github_token", "elapsed_seconds": time.time() - start}
    evaluator = Evaluator(
        pipeline,
        api_key=pipeline.config.llm_api_key,
        model=pipeline.config.model,
        provider=pipeline.config.provider,
    )
    if not human_comments:
        human_comments = pipeline.github.get_review_ground_truth(repo_name, pr_number)
    if not human_comments:
        return {
            "status": "skipped",
            "reason": "no_human_review_comments",
            "elapsed_seconds": time.time() - start,
        }
    old_max = pipeline.config.max_comments
    pipeline.config.max_comments = max(1, int(max_comments_per_pr))
    comments = pipeline.review_pr(
        repo_name,
        pr_number,
        target_dir=target_dir,
        snapshot_mode=snapshot_mode,
    )
    pipeline.config.max_comments = old_max
    review_summary = dict(getattr(pipeline, "last_review_summary", {}) or {})
    if not comments:
        reason = collapse_reason(review_summary)
        return {
            "status": "skipped",
            "reason": reason,
            "review_summary": review_summary,
            "elapsed_seconds": time.time() - start,
        }
    # Only evaluate LLM-generated comments for address rate.
    # Static findings (ruff/semgrep) rarely match human reviews and dilute metrics.
    llm_comments = [c for c in comments if c.get("source") != "static"]
    outcomes = []
    for comment in llm_comments:
        best = evaluator._best_human_match(comment, human_comments)
        matched = best is not None
        addressed = bool(best.get("addressed", False)) if best is not None else False
        plausible = (not matched) and evaluator._is_plausible(comment)
        outcomes.append(
            {
                "matched": matched,
                "addressed": addressed,
                "plausible": plausible,
                "verified": bool(comment.get("suggested_code")),
            }
        )
    return {
        "status": "ok",
        "pr_number": pr_number,
        "comment_outcomes": outcomes,
        "review_summary": review_summary,
        "elapsed_seconds": time.time() - start,
    }


def collapse_reason(review_summary: dict) -> str:
    if not review_summary:
        return "no_generated_comments"
    if review_summary.get("reviewable_changed_file_count", 0) > 0 and review_summary.get("changed_node_count", 0) == 0:
        return "no_indexed_functions_in_reviewable_files"
    if review_summary.get("target_count", 0) == 0:
        return "no_targets"
    if review_summary.get("survivor_count", 0) == 0:
        return "review_loop_zero_survivors"
    if review_summary.get("ranked_count_before_bundle", 0) == 0:
        return "ranker_zero_comments"
    if review_summary.get("final_comment_count", 0) == 0 and review_summary.get("bundle_drop_count", 0) > 0:
        return "bundle_dropped_all"
    return "no_generated_comments"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
