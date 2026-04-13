from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from ares.agents import Critic, Investigator, Reviewer, Verifier
from ares.config import AresConfig
from ares.evaluate import Evaluator
from ares.feedback import FeedbackCollector, FeedbackLearner, ReviewStrategy
from ares.graph import GraphQuery, RepositoryIndexer
from ares.graph.classifier import NodeClassifier
from ares.integrations import GitHubClient, Neo4jClient, PineconeClient
from ares.ranker import Ranker
from ares.review_scope import reviewable_source_files
from ares.static_analysis import StaticAnalyzer

VERIFY_SKIP_THRESHOLD = 0.3   # only skip clearly junk candidates (was 0.6)
VERIFY_FULL_THRESHOLD = 0.75  # full verify earlier; compile-only for mid-range (was 0.85)
EARLY_CRITIQUE_FLOOR = 0.4    # drop candidates below this before expensive verification


def _log(pr_number: int, step: str, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    label = f"PR #{pr_number}" if pr_number else "    "
    print(f"[timing] {label}  {step}: {elapsed:.2f}s", file=sys.stderr, flush=True)


class Pipeline:
    def __init__(self, config: AresConfig):
        self.config = config
        self.loaded_graph_repo = ""
        self._active_repo_name = ""  # e.g. "fastapi/fastapi" — Neo4j namespace
        self.github = GitHubClient(config.github_token) if config.github_token else None
        self.pinecone = PineconeClient(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            namespace=config.pinecone_namespace,
        )
        
        self.neo4j = Neo4jClient(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
        )
        strategy_root = config.repo_path or config.workspace_root
        self.strategy = ReviewStrategy.load(strategy_root)
        self.collector = FeedbackCollector(
            strategy_root,
            github_client=self.github,
            pinecone_client=self.pinecone,
        )
        self.learner = FeedbackLearner(
            strategy_root,
            strategy=self.strategy,
            api_key=config.llm_api_key,
            model=config.model,
            provider=config.provider,
        )
        self.static = StaticAnalyzer(config.repo_path) if config.repo_path else None
        self.graph = None
        self.graph_query = None
        self.investigator = None
        self.reviewer = None
        self.verifier = Verifier(
            config.llm_api_key,
            config.repo_path or os.getcwd(),
            model=config.lightweight_model,
            provider=config.provider,
        )
        self.critic = None
        self.ranker = None
        self.last_review_loop: dict = {"passes": [], "best_average_score": 0.0}
        self.last_review_summary: dict = {}
        self._rebuild_strategy_stages()
        if config.repo_path:
            self._activate_repo(config.repo_path)

    def index_repo(self, repo_path: str, repo_name: str = "") -> None:
        self._activate_repo(repo_path)
        indexer = RepositoryIndexer(
            self.config.repo_path,
            strategy=self.strategy,
            neo4j_client=self.neo4j,
            repo_name=repo_name or self._active_repo_name,
        )
        graph = indexer.build(include_git_metadata=bool(indexer._neo4j))
        if indexer._neo4j:
            indexer.save(graph)
        self.graph = graph
        self.loaded_graph_repo = self.config.repo_path
        self.graph_query = self._make_graph_query()
        self.investigator = Investigator(self.graph_query, self.config.repo_path)

    def load_graph(self, repo_path: str, repo_name: str = "") -> None:
        self._activate_repo(repo_path)
        indexer = RepositoryIndexer(
            repo_path,
            strategy=self.strategy,
            neo4j_client=self.neo4j,
            repo_name=repo_name or self._active_repo_name,
        )
        self.graph = indexer.load()
        self.loaded_graph_repo = os.path.abspath(repo_path)
        self.graph_query = self._make_graph_query()
        self.investigator = Investigator(self.graph_query, repo_path)

    def review_pr(
        self,
        repo_name: str,
        pr_number: int,
        target_dir: str | None = None,
        snapshot_mode: str = "final",
    ) -> list[dict]:
        if self.github is None:
            raise RuntimeError("GITHUB_TOKEN is required for PR review.")
        self._active_repo_name = repo_name
        _t0 = time.perf_counter()
        pr_data = self.github.get_pr_data(
            repo_name,
            pr_number,
            snapshot_mode=snapshot_mode,
        )
        _log(pr_number, "get_pr_data", _t0)

        clone_dir = target_dir or os.path.join(
            self.config.workspace_root,
            repo_name.replace("/", "__"),
        )
        fetch_ref = pr_data.get("fetch_ref") or pr_data.get("head_ref") or pr_data["head_branch"]
        checkout_sha = pr_data.get("checkout_sha")
        _t = time.perf_counter()
        clone_depth = int(os.getenv("ARES_CLONE_DEPTH", "1"))
        if hasattr(self.github, "clone_repo_ref"):
            self.github.clone_repo_ref(
                repo_name,
                fetch_ref,
                clone_dir,
                checkout_name=f"ares-pr-{pr_number}",
                checkout_sha=checkout_sha,
                depth=clone_depth,
            )
        else:
            self.github.clone_repo(repo_name, pr_data["head_branch"], clone_dir, depth=clone_depth)
        _log(pr_number, "clone", _t)

        self._activate_repo(clone_dir)
        _t = time.perf_counter()
        if not self._graph_exists(clone_dir):
            if self.neo4j.available and self._active_repo_name:
                try:
                    self._patch_graph_for_pr(clone_dir, pr_data["diff"], repo_name=repo_name)
                    _log(pr_number, "patch_graph_for_pr", _t)
                except (FileNotFoundError, Exception) as exc:
                    if "No graph found" in str(exc):
                        _log(pr_number, f"no base graph in Neo4j, full indexing")
                        _t = time.perf_counter()
                        self.index_repo(clone_dir, repo_name=repo_name)
                        _log(pr_number, "index_repo_full (fallback)", _t)
                    else:
                        raise
            else:
                self.index_repo(clone_dir, repo_name=repo_name)
                _log(pr_number, "index_repo_full", _t)
        elif self.graph is None or self.loaded_graph_repo != os.path.abspath(clone_dir):
            self.load_graph(clone_dir, repo_name=repo_name)
            _log(pr_number, "load_graph", _t)
        else:
            _log(pr_number, "graph_already_loaded", _t)

        _t = time.perf_counter()
        reviewable_changed_files = reviewable_source_files(pr_data["changed_files"])
        # Parse diff hunks once; derive ranges for node mapping and pass hunks to investigator.
        diff_hunks = self.investigator._parse_diff_hunks(pr_data["diff"])
        changed_ranges = {
            filepath: [(h["start"], h["end"]) for h in hunks]
            for filepath, hunks in diff_hunks.items()
        }
        changed_nodes = self._map_diff_to_nodes(pr_data["diff"], reviewable_changed_files, changed_ranges=changed_ranges)
        self._enrich_review_nodes(changed_nodes)
        _log(pr_number, f"map_and_enrich ({len(changed_nodes)} nodes)", _t)

        description = "\n".join(
            [pr_data["title"], pr_data["description"], *pr_data.get("commit_messages", [])]
        )
        _t = time.perf_counter()
        static_findings = self.static.analyze_changed_files(pr_data["changed_files"])
        _log(pr_number, f"static_analysis ({len(static_findings)} findings)", _t)

        _t = time.perf_counter()
        targets = self.investigator.investigate(
            pr_data["diff"],
            description,
            changed_nodes,
            changed_files=pr_data["changed_files"],
            reviewable_changed_files=reviewable_changed_files,
            diff_hunks=diff_hunks,
        )
        _log(pr_number, f"investigate ({len(targets)} targets)", _t)

        critique_context = targets[0]["pr_intent"] if targets else description
        file_sources = self._collect_file_sources(clone_dir, pr_data["changed_files"])
        _t = time.perf_counter()
        survivors, candidate_scores, pass_metrics = self._run_review_repl(
            targets,
            critique_context,
            file_sources,
        )
        _log(pr_number, f"review_repl ({len(survivors)} survivors)", _t)
        ranked_comments = self.ranker.rank_and_cap(static_findings, survivors)
        final_comments = self._apply_bundle_verification(ranked_comments, file_sources)
        self.last_review_summary = {
            "repo_name": repo_name,
            "pr_number": pr_number,
            "changed_node_count": len(changed_nodes),
            "reviewable_changed_file_count": len(reviewable_changed_files),
            "static_finding_count": len(static_findings),
            "target_count": len(targets),
            "survivor_count": len(survivors),
            "ranked_count_before_bundle": len(ranked_comments),
            "final_comment_count": len(final_comments),
            "bundle_drop_count": max(0, len(ranked_comments) - len(final_comments)),
            "review_loop": self.last_review_loop,
        }
        self._attach_review_loop_metadata(final_comments, pass_metrics, candidate_scores)
        self.github.post_review_comments(repo_name, pr_number, final_comments)
        self.collector.record_posted_comments(repo_name, pr_number, final_comments)
        return final_comments

    def learn(self, repo_path: str) -> dict:
        self._activate_repo(repo_path)
        collected = self.collector.collect_feedback()
        learned = self.learner.improve()
        self.strategy = ReviewStrategy.load(self.config.repo_path)
        self._rebuild_strategy_stages()
        if self.graph is not None and self.loaded_graph_repo == self.config.repo_path:
            self.graph_query = self._make_graph_query()
            self.investigator = Investigator(self.graph_query, self.config.repo_path)
        return {"collector": collected, "learner": learned}

    def _enrich_review_nodes(self, changed_nodes: list[str]) -> None:
        if not changed_nodes or self.graph is None:
            return
        NodeClassifier(
            self.graph,
            self.config.repo_path,
            strategy=self.strategy,
            neo4j_client=self.neo4j,
            repo_name=self._active_repo_name,
        ).enrich_nodes(changed_nodes)

    def _graph_exists(self, repo_path: str) -> bool:
        return self.neo4j.available and self._active_repo_name and self.neo4j.has_graph(self._active_repo_name)

    def _patch_graph_for_pr(self, clone_dir: str, diff: str, repo_name: str = "") -> None:
        """Load the existing graph from Neo4j and patch only the files changed in the PR."""
        _t = time.perf_counter()
        self.load_graph(clone_dir, repo_name=repo_name)
        _log(0, "  patch: load_graph from Neo4j", _t)
        changed_files = list(self._parse_changed_ranges(diff).keys())
        if changed_files:
            _t = time.perf_counter()
            indexer = RepositoryIndexer(
                clone_dir,
                strategy=self.strategy,
                neo4j_client=self.neo4j,
                repo_name=repo_name or self._active_repo_name,
            )
            self.graph = indexer.patch_files(self.graph, changed_files)
            # patch_files already saves changed nodes to Neo4j incrementally;
            # no need for a redundant full-graph save here.
            _log(0, f"  patch: patch_files ({len(changed_files)} files)", _t)
        self.loaded_graph_repo = os.path.abspath(clone_dir)

    def _make_graph_query(self) -> GraphQuery:
        return GraphQuery(self.graph, strategy=self.strategy, neo4j_client=self.neo4j, repo_name=self._active_repo_name)

    def _activate_repo(self, repo_path: str) -> None:
        self.config.repo_path = os.path.abspath(repo_path)
        self.strategy = ReviewStrategy.load(self.config.repo_path)
        self.collector.set_repo_path(self.config.repo_path)
        self.learner.set_repo_path(self.config.repo_path)
        self.learner.strategy = self.strategy
        self.static = StaticAnalyzer(self.config.repo_path)
        self.verifier.repo_path = self.config.repo_path
        self._rebuild_strategy_stages()
        if self.graph is not None and self.loaded_graph_repo == self.config.repo_path:
            self.graph_query = self._make_graph_query()
            self.investigator = Investigator(self.graph_query, self.config.repo_path)

    def _rebuild_strategy_stages(self) -> None:
        self.reviewer = Reviewer(
            self.config.llm_api_key,
            model=self.config.model,
            provider=self.config.provider,
            extra_instructions=self.strategy.extra_reviewer_instructions,
            pinecone_index=self.pinecone,
        )
        self.critic = Critic(
            self.config.llm_api_key,
            pinecone_index=self.pinecone,
            model=self.config.model,
            provider=self.config.provider,
            strategy=self.strategy,
            lightweight_model=self.config.lightweight_model,
        )
        self.ranker = Ranker(max_comments=self.config.max_comments, strategy=self.strategy)

    def _collect_file_sources(self, repo_path: str, changed_files: list[str]) -> dict[str, str]:
        sources = {}
        for relative_path in changed_files:
            file_path = Path(repo_path) / relative_path
            if file_path.exists():
                sources[relative_path] = file_path.read_text(encoding="utf-8", errors="ignore")
        return sources

    def _run_review_repl(
        self,
        targets: list[dict],
        critique_context: str,
        file_sources: dict[str, str] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        if not targets:
            self.last_review_loop = {"passes": [], "best_average_score": 0.0}
            return [], [], []
        current_targets = [dict(target) for target in targets]
        pass_metrics: list[dict] = []
        best_candidates: list[dict] = []
        best_scores: list[dict] = []
        best_average = 0.0
        previous_average: float | None = None
        verification_cache: dict[str, dict] = {}
        score_cache: dict[str, dict] = {}
        max_passes = max(1, int(self.config.max_review_passes))
        for pass_number in range(1, max_passes + 1):
            _tp = time.perf_counter()
            n_agg = self.config.review_aggregation_runs
            if n_agg > 1 and pass_number == 1:
                candidates = self._aggregate_multi_reviews(current_targets, n_runs=n_agg)
            else:
                candidates = self.reviewer.review(current_targets)
            _log(0, f"  pass {pass_number} reviewer ({len(candidates)} candidates)", _tp)

            pre_scores = self._prescore_candidates(candidates, critique_context)
            routed_candidates = self._attach_prescore_metadata(candidates, pre_scores)
            verification_modes = self._build_verification_modes(pre_scores)

            _tp = time.perf_counter()
            routed_candidates, verification_modes = self._early_critique_filter(
                routed_candidates, verification_modes, pre_scores,
            )
            _log(0, f"  pass {pass_number} early_critique ({len(routed_candidates)} survivors)", _tp)

            if self.config.actionability_filter and routed_candidates:
                _tp = time.perf_counter()
                routed_candidates, verification_modes = self._actionability_filter(
                    routed_candidates, verification_modes,
                )
                _log(0, f"  pass {pass_number} actionability ({len(routed_candidates)} survivors)", _tp)

            _tp = time.perf_counter()
            verified_candidates = self.verifier.verify_candidates(
                routed_candidates,
                file_sources or {},
                cache=verification_cache,
                verification_modes=verification_modes,
            )
            _log(0, f"  pass {pass_number} verifier (modes={verification_modes})", _tp)

            _tp = time.perf_counter()
            scores = self.critic.score_comments(verified_candidates, critique_context, score_cache=score_cache)
            _log(0, f"  pass {pass_number} critic", _tp)
            selected_candidates = self.critic.select_comments(verified_candidates, scores)
            average_score = self.critic.average_score(scores)
            status_counts = {"passed": 0, "inconclusive": 0, "disproved": 0, "skipped": 0}
            for candidate in verified_candidates:
                status = candidate.get("verification", {}).get("status", "inconclusive")
                status_counts[status] = status_counts.get(status, 0) + 1
            metric = {
                "pass": pass_number,
                "candidate_count": len(candidates),
                "skipped_verification_count": verification_modes.count("skip"),
                "lightweight_verification_count": verification_modes.count("compile"),
                "full_verification_count": verification_modes.count("full"),
                "verified_count": status_counts.get("passed", 0),
                "inconclusive_count": status_counts.get("inconclusive", 0),
                "disproved_count": status_counts.get("disproved", 0),
                "skipped_count": status_counts.get("skipped", 0),
                "average_score": round(average_score, 3),
                "high_score_count": sum(1 for score in scores if score.get("keep")),
            }
            pass_metrics.append(metric)
            if average_score >= best_average:
                best_average = average_score
                best_candidates = selected_candidates
                best_scores = scores
            all_selected_passed = bool(selected_candidates) and all(
                candidate.get("verification", {}).get("status") == "passed"
                for candidate in selected_candidates
            )
            all_verified_high = all_selected_passed and self.critic.all_scores_high(scores) and all(
                candidate.get("verification", {}).get("status") == "passed"
                for candidate in verified_candidates
            )
            if not candidates or not scores or all_verified_high:
                break
            if previous_average is not None and average_score <= previous_average + 0.05:
                break
            if pass_number == max_passes:
                break
            current_targets = self.reviewer.refine(current_targets, verified_candidates, scores)
            previous_average = average_score
        self.last_review_loop = {
            "passes": pass_metrics,
            "best_average_score": round(best_average, 3),
        }
        return best_candidates, best_scores, pass_metrics

    def _early_critique_filter(
        self,
        candidates: list[dict],
        verification_modes: list[str],
        pre_scores: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """Use existing prescores to drop weak candidates before expensive verification."""
        needs_verification = [
            i for i, mode in enumerate(verification_modes) if mode != "skip"
        ]
        if not needs_verification:
            return candidates, verification_modes
        keep_indices: set[int] = set()
        for idx in needs_verification:
            score = pre_scores[idx] if idx < len(pre_scores) else {}
            if float(score.get("score", 0.0)) >= EARLY_CRITIQUE_FLOOR:
                keep_indices.add(idx)
        skip_indices = {i for i in range(len(candidates)) if verification_modes[i] == "skip"}
        surviving = []
        surviving_modes = []
        for i in range(len(candidates)):
            if i in skip_indices or i in keep_indices:
                surviving.append(candidates[i])
                surviving_modes.append(verification_modes[i])
        return surviving, surviving_modes

    def _actionability_filter(
        self,
        candidates: list[dict],
        verification_modes: list[str],
    ) -> tuple[list[dict], list[str]]:
        """Drop non-actionable candidates before expensive verification."""
        classifications = self.critic.classify_actionability(candidates)
        surviving = []
        surviving_modes = []
        for i, (candidate, mode) in enumerate(zip(candidates, verification_modes)):
            cl = classifications[i] if i < len(classifications) else {"actionable": True}
            if cl.get("actionable", True):
                surviving.append(candidate)
                surviving_modes.append(mode)
            else:
                print(
                    f"[actionability] dropped: {candidate.get('file', '?')}:"
                    f"{candidate.get('line_start', '?')} -- {cl.get('reason', '')}",
                    flush=True,
                )
        return surviving, surviving_modes

    def _aggregate_multi_reviews(
        self,
        targets: list[dict],
        n_runs: int = 3,
    ) -> list[dict]:
        """Run reviewer N times with temperature variation and keep consensus.

        Candidates appearing in ≥2 independent runs (by semantic similarity
        clustering) survive. The best-worded version from each cluster is kept.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ares.utils.text_similarity import batch_similarities

        temperatures = [0.0, 0.3, 0.4, 0.5, 0.6][:n_runs]
        all_candidates: list[dict] = []

        def _run_review(run_idx: int) -> tuple[int, list[dict], float]:
            temp = temperatures[run_idx]
            _tp = time.perf_counter()
            candidates = self.reviewer.review(targets, temperature=temp)
            elapsed = time.perf_counter() - _tp
            for c in candidates:
                c["_agg_run"] = run_idx
            return run_idx, candidates, elapsed

        # Run all aggregation reviews in parallel
        _tp_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_runs) as pool:
            futures = [pool.submit(_run_review, i) for i in range(n_runs)]
            for future in as_completed(futures):
                run_idx, candidates, elapsed = future.result()
                temp = temperatures[run_idx]
                print(
                    f"[timing]         aggregation run {run_idx + 1}/{n_runs} "
                    f"(temp={temp:.1f}, {len(candidates)} candidates): {elapsed:.2f}s",
                    file=sys.stderr, flush=True,
                )
                all_candidates.extend(candidates)

        if not all_candidates:
            return []

        # Cluster by (file, nearby lines, text similarity)
        clusters: list[list[int]] = []
        assigned = [False] * len(all_candidates)

        for i, ci in enumerate(all_candidates):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            ci_file = ci.get("file", "")
            ci_start = ci.get("line_start", 0)
            ci_end = ci.get("line_end", 0)
            ci_comment = ci.get("comment", "")

            # Find candidates from other runs in same file/line vicinity
            nearby_indices = []
            nearby_texts = []
            for j, cj in enumerate(all_candidates):
                if assigned[j] or j == i:
                    continue
                if cj.get("file", "") != ci_file:
                    continue
                cj_start = cj.get("line_start", 0)
                cj_end = cj.get("line_end", 0)
                if cj_end < ci_start - 5 or cj_start > ci_end + 5:
                    continue
                nearby_indices.append(j)
                nearby_texts.append(cj.get("comment", ""))

            if nearby_texts:
                sims = batch_similarities(ci_comment, nearby_texts)
                for k, sim in enumerate(sims):
                    j = nearby_indices[k]
                    if sim >= 0.7 and not assigned[j]:
                        cluster.append(j)
                        assigned[j] = True

            clusters.append(cluster)

        # Keep clusters with ≥2 distinct runs (consensus)
        consensus = []
        for cluster in clusters:
            runs_in_cluster = {all_candidates[idx].get("_agg_run", -1) for idx in cluster}
            if len(runs_in_cluster) < 2:
                continue
            # Pick highest-confidence version
            best_idx = max(cluster, key=lambda idx: float(all_candidates[idx].get("confidence", 0.0)))
            candidate = dict(all_candidates[best_idx])
            candidate["consensus_count"] = len(runs_in_cluster)
            candidate["consensus_agreement"] = round(len(runs_in_cluster) / n_runs, 2)
            candidate.pop("_agg_run", None)
            consensus.append(candidate)

        print(
            f"[timing]         aggregation consensus: {len(consensus)} from "
            f"{len(all_candidates)} raw candidates",
            file=sys.stderr, flush=True,
        )
        return consensus

    def _prescore_candidates(self, candidates: list[dict], critique_context: str) -> list[dict]:
        if not candidates:
            return []
        return self.critic.prescore_comments(candidates, critique_context)

    def _build_verification_modes(self, scores: list[dict]) -> list[str]:
        modes = []
        for score in scores:
            value = float(score.get("score", 0.0))
            if value < VERIFY_SKIP_THRESHOLD:
                modes.append("skip")
            elif value < VERIFY_FULL_THRESHOLD:
                modes.append("compile")
            else:
                modes.append("full")
        return modes

    def _attach_prescore_metadata(self, candidates: list[dict], scores: list[dict]) -> list[dict]:
        enriched = []
        for candidate, score in zip(candidates, scores):
            enriched.append(
                {
                    **candidate,
                    "pre_verification_score": float(score.get("score", 0.0)),
                    "pre_verification_reason": score.get("reason", ""),
                }
            )
        return enriched

    def _apply_bundle_verification(self, comments: list[dict], file_sources: dict[str, str]) -> list[dict]:
        if not comments:
            return []
        eligible = []
        key_to_result: dict[str, dict] = {}
        for comment in comments:
            if comment.get("source") != "llm":
                key = self._comment_bundle_key(comment)
                key_to_result[key] = {
                    **comment,
                    "bundle_verification": {
                        "status": "skipped",
                        "reason": "Bundle verification only applies to LLM comments with a synthesized fix.",
                        "group_size": 0,
                    },
                }
                continue
            if not comment.get("suggested_code") or not comment.get("validation", {}).get("compiles", False):
                key = self._comment_bundle_key(comment)
                key_to_result[key] = {
                    **comment,
                    "bundle_verification": {
                        "status": "skipped",
                        "reason": "Comment did not have a compilable synthesized fix for bundle verification.",
                        "group_size": 0,
                    },
                }
                continue
            eligible.append(comment)
        if len(eligible) <= 1:
            for comment in eligible:
                key_to_result[self._comment_bundle_key(comment)] = {
                    **comment,
                    "bundle_verification": {
                        "status": "skipped",
                        "reason": "Bundle verification requires at least two eligible LLM comments.",
                        "group_size": len(eligible),
                    },
                }
        else:
            result = self.verifier.verify_bundle(eligible, file_sources)
            for comment in result.get("survivors", []):
                key_to_result[self._comment_bundle_key(comment)] = comment
            dropped_keys = {
                self._comment_bundle_key(comment): comment
                for comment in result.get("dropped", [])
            }
            for key, comment in dropped_keys.items():
                key_to_result[key] = comment
        filtered: list[dict] = []
        for comment in comments:
            key = self._comment_bundle_key(comment)
            resolved = key_to_result.get(key, comment)
            if resolved.get("bundle_verification", {}).get("status") == "failed":
                continue
            filtered.append(resolved)
        return filtered

    def _comment_bundle_key(self, comment: dict) -> str:
        return str(comment.get("verification_key") or f"{comment.get('node_id', '')}|{comment.get('comment', '')}")

    def _attach_review_loop_metadata(
        self,
        comments: list[dict],
        pass_metrics: list[dict],
        scores: list[dict],
    ) -> None:
        if not comments:
            return
        score_by_key = {
            (
                score.get("node_id", ""),
                score.get("comment", ""),
            ): score
            for score in scores
        }
        loop_summary = {
            "passes": pass_metrics,
            "best_average_score": self.last_review_loop.get("best_average_score", 0.0),
        }
        for comment in comments:
            critique = score_by_key.get((comment.get("node_id", ""), comment.get("comment", "")), {})
            comment["review_loop"] = loop_summary
            if critique:
                comment["critic_score"] = critique.get("score", 0.0)
                comment["critique_reason"] = critique.get("reason", "")

    # ------------------------------------------------------------------
    # Batch API orchestration — processes N PRs with batched LLM calls
    # for 50% cost savings via Anthropic's Batch API.
    # ------------------------------------------------------------------

    def batch_review_prs(
        self,
        pr_configs: list[dict],
    ) -> list[list[dict]]:
        """Review multiple PRs using the Anthropic Batch API (50% cost discount).

        *pr_configs* is a list of dicts, each with keys ``repo_name``,
        ``pr_number``, and optionally ``target_dir``.

        Returns a list of comment lists, one per input PR (same order).
        """
        n = len(pr_configs)
        if not n:
            return []

        # ---------- Phase 1: investigate all PRs (no LLM) ----------
        pr_states: list[dict] = []
        for pr_cfg in pr_configs:
            state = self._batch_investigate_pr(pr_cfg)
            pr_states.append(state)

        # ---------- Phase 2: batched multi-pass review ----------
        max_passes = max(1, int(self.config.max_review_passes))
        # Per-PR mutable state
        for st in pr_states:
            st["current_targets"] = [dict(t) for t in st["targets"]]
            st["best_candidates"] = []
            st["best_scores"] = []
            st["best_average"] = 0.0
            st["previous_average"] = None
            st["verification_cache"] = {}
            st["score_cache"] = {}
            st["active"] = bool(st["targets"])
            st["pass_metrics"] = []

        for pass_number in range(1, max_passes + 1):
            active_indices = [i for i in range(n) if pr_states[i]["active"]]
            if not active_indices:
                break

            # -- Step A: batch reviewer calls --
            review_requests: list[tuple[str, str, int]] = []
            request_map: list[int] = []  # maps request index → pr index
            for i in active_indices:
                targets = pr_states[i]["current_targets"]
                # May produce multiple requests if targets > _BATCH_SIZE
                batches: list[list[dict]] = []
                for b in range(0, len(targets), 12):
                    batches.append(targets[b : b + 12])
                for batch in batches:
                    req = self.reviewer.build_review_request(batch)
                    review_requests.append(req)
                    request_map.append(i)
                pr_states[i]["_review_batches"] = batches

            review_responses = self.reviewer.client.complete_batch(review_requests)

            # Parse responses back per PR
            resp_idx = 0
            for i in active_indices:
                all_candidates: list[dict] = []
                for batch in pr_states[i]["_review_batches"]:
                    raw = review_responses[resp_idx]
                    all_candidates.extend(self.reviewer.parse_review_response(batch, raw))
                    resp_idx += 1
                pr_states[i]["candidates"] = all_candidates

            # -- Step B: prescore + early critique (heuristic, no LLM) --
            for i in active_indices:
                candidates = pr_states[i]["candidates"]
                critique_context = pr_states[i]["critique_context"]
                pre_scores = self._prescore_candidates(candidates, critique_context)
                routed = self._attach_prescore_metadata(candidates, pre_scores)
                modes = self._build_verification_modes(pre_scores)
                routed, modes = self._early_critique_filter(routed, modes, pre_scores)
                pr_states[i]["routed_candidates"] = routed
                pr_states[i]["verification_modes"] = modes

            # -- Step C: batch verifier fix generation --
            fix_requests_all: list[tuple[str, str, int]] = []
            fix_request_meta: list[tuple[int, list[tuple], list[tuple[list[tuple], str, str, int]]]] = []
            for i in active_indices:
                routed = pr_states[i]["routed_candidates"]
                modes = pr_states[i]["verification_modes"]
                cache = pr_states[i]["verification_cache"]
                _, pending = self.verifier.prepare_pending(routed, modes, cache)
                pr_states[i]["pending"] = pending
                pr_states[i]["full_pending"] = [item for item in pending if item[3] == "full"]
                fix_reqs = self.verifier.build_fix_batch_requests(pending)
                pr_states[i]["fix_reqs"] = fix_reqs
                for (_batch, sys_p, usr_p, max_t) in fix_reqs:
                    fix_requests_all.append((sys_p, usr_p, max_t))
                fix_request_meta.append((i, pending, fix_reqs))

            if fix_requests_all:
                fix_responses = self.verifier.client.complete_batch(fix_requests_all)
            else:
                fix_responses = []

            # Distribute fix responses back per PR
            fix_resp_idx = 0
            for i in active_indices:
                fix_reqs = pr_states[i]["fix_reqs"]
                pr_fix_responses = []
                for _ in fix_reqs:
                    pr_fix_responses.append(fix_responses[fix_resp_idx] if fix_resp_idx < len(fix_responses) else "")
                    fix_resp_idx += 1
                pr_states[i]["pre_fixes"] = self.verifier.parse_fix_batch_responses(fix_reqs, pr_fix_responses)

            # -- Step D: batch verifier repro test generation --
            repro_requests_all: list[tuple[str, str, int]] = []
            for i in active_indices:
                full_pending = pr_states[i]["full_pending"]
                repro_reqs = self.verifier.build_repro_batch_requests(full_pending)
                pr_states[i]["repro_reqs"] = repro_reqs
                for (_batch, _meta, sys_p, usr_p, max_t) in repro_reqs:
                    repro_requests_all.append((sys_p, usr_p, max_t))

            if repro_requests_all:
                repro_responses = self.verifier.client.complete_batch(repro_requests_all)
            else:
                repro_responses = []

            repro_resp_idx = 0
            for i in active_indices:
                repro_reqs = pr_states[i]["repro_reqs"]
                pr_repro_responses = []
                for _ in repro_reqs:
                    pr_repro_responses.append(repro_responses[repro_resp_idx] if repro_resp_idx < len(repro_responses) else "")
                    repro_resp_idx += 1
                pr_states[i]["pre_repros"] = self.verifier.parse_repro_batch_responses(repro_reqs, pr_repro_responses)

            # -- Step E: run local verification per PR (CPU, no LLM) --
            for i in active_indices:
                verified = self.verifier.verify_candidates(
                    pr_states[i]["routed_candidates"],
                    pr_states[i]["file_sources"],
                    cache=pr_states[i]["verification_cache"],
                    verification_modes=pr_states[i]["verification_modes"],
                    pre_fixes=pr_states[i]["pre_fixes"],
                    pre_repros=pr_states[i]["pre_repros"],
                )
                pr_states[i]["verified_candidates"] = verified

            # -- Step F: batch critic scoring --
            score_requests_all: list[tuple[str, str, int]] = []
            score_pr_indices: list[int] = []
            for i in active_indices:
                verified = pr_states[i]["verified_candidates"]
                critique_context = pr_states[i]["critique_context"]
                req = self.critic.build_score_request(verified, critique_context)
                if req is not None:
                    score_requests_all.append(req)
                    score_pr_indices.append(i)

            if score_requests_all:
                score_responses = self.critic.lightweight_client.complete_batch(score_requests_all)
            else:
                score_responses = []

            for resp_i, i in enumerate(score_pr_indices):
                raw = score_responses[resp_i] if resp_i < len(score_responses) else ""
                verified = pr_states[i]["verified_candidates"]
                scores = self.critic.parse_score_response(verified, raw)
                if not scores:
                    scores = self.critic.prescore_comments(verified, pr_states[i]["critique_context"])
                pr_states[i]["scores"] = scores

            # For active PRs without a score response (e.g. no verified candidates)
            for i in active_indices:
                if "scores" not in pr_states[i]:
                    pr_states[i]["scores"] = self.critic.prescore_comments(
                        pr_states[i].get("verified_candidates", []),
                        pr_states[i]["critique_context"],
                    )

            # -- Step G: select and check exit conditions per PR --
            for i in active_indices:
                verified = pr_states[i]["verified_candidates"]
                scores = pr_states[i]["scores"]
                selected = self.critic.select_comments(verified, scores)
                average = self.critic.average_score(scores)

                pr_states[i]["pass_metrics"].append({
                    "pass": pass_number,
                    "candidate_count": len(pr_states[i]["candidates"]),
                    "average_score": round(average, 3),
                })

                if average >= pr_states[i]["best_average"]:
                    pr_states[i]["best_average"] = average
                    pr_states[i]["best_candidates"] = selected
                    pr_states[i]["best_scores"] = scores

                should_stop = (
                    not pr_states[i]["candidates"]
                    or not scores
                    or self.critic.all_scores_high(scores)
                )
                if not should_stop and pr_states[i]["previous_average"] is not None:
                    if average <= pr_states[i]["previous_average"] + 0.05:
                        should_stop = True
                if should_stop or pass_number == max_passes:
                    pr_states[i]["active"] = False
                else:
                    pr_states[i]["current_targets"] = self.reviewer.refine(
                        pr_states[i]["current_targets"], verified, scores,
                    )
                    pr_states[i]["previous_average"] = average

        # ---------- Phase 3: rank, bundle-verify, return ----------
        all_results: list[list[dict]] = []
        for i, st in enumerate(pr_states):
            survivors = st.get("best_candidates", [])
            file_sources = st.get("file_sources", {})
            static_findings = st.get("static_findings", [])
            ranked = self.ranker.rank_and_cap(static_findings, survivors)
            final = self._apply_bundle_verification(ranked, file_sources)
            all_results.append(final)

        return all_results

    def _batch_investigate_pr(self, pr_cfg: dict) -> dict:
        """Run the non-LLM investigation phase for a single PR.

        Returns a state dict with targets, file_sources, etc.
        """
        repo_name = pr_cfg["repo_name"]
        pr_number = pr_cfg["pr_number"]
        target_dir = pr_cfg.get("target_dir")

        self._active_repo_name = repo_name
        pr_data = self.github.get_pr_data(repo_name, pr_number)

        clone_dir = target_dir or os.path.join(
            self.config.workspace_root, repo_name.replace("/", "__"),
        )
        clone_ref = pr_data.get("head_ref") or pr_data["head_branch"]
        clone_depth = int(os.getenv("ARES_CLONE_DEPTH", "1"))
        if hasattr(self.github, "clone_repo_ref"):
            self.github.clone_repo_ref(repo_name, clone_ref, clone_dir, checkout_name=f"ares-pr-{pr_number}", depth=clone_depth)
        else:
            self.github.clone_repo(repo_name, pr_data["head_branch"], clone_dir, depth=clone_depth)

        self._activate_repo(clone_dir)
        if not self._graph_exists(clone_dir):
            if self.neo4j.available and self._active_repo_name:
                self._patch_graph_for_pr(clone_dir, pr_data["diff"], repo_name=repo_name)
            else:
                self.index_repo(clone_dir, repo_name=repo_name)
        elif self.graph is None or self.loaded_graph_repo != os.path.abspath(clone_dir):
            self.load_graph(clone_dir, repo_name=repo_name)

        reviewable_changed_files = reviewable_source_files(pr_data["changed_files"])
        diff_hunks = self.investigator._parse_diff_hunks(pr_data["diff"])
        changed_ranges = {
            filepath: [(h["start"], h["end"]) for h in hunks]
            for filepath, hunks in diff_hunks.items()
        }
        changed_nodes = self._map_diff_to_nodes(pr_data["diff"], reviewable_changed_files, changed_ranges=changed_ranges)
        self._enrich_review_nodes(changed_nodes)

        description = "\n".join([pr_data["title"], pr_data["description"], *pr_data.get("commit_messages", [])])
        static_findings = self.static.analyze_changed_files(pr_data["changed_files"])

        targets = self.investigator.investigate(
            pr_data["diff"], description, changed_nodes,
            changed_files=pr_data["changed_files"],
            reviewable_changed_files=reviewable_changed_files,
            diff_hunks=diff_hunks,
        )
        critique_context = targets[0]["pr_intent"] if targets else description
        file_sources = self._collect_file_sources(clone_dir, pr_data["changed_files"])

        return {
            "repo_name": repo_name,
            "pr_number": pr_number,
            "pr_data": pr_data,
            "targets": targets,
            "critique_context": critique_context,
            "file_sources": file_sources,
            "static_findings": static_findings,
            "clone_dir": clone_dir,
        }

    def _functions_in_file(self, filepath: str) -> list[str]:
        """Return function node IDs for *filepath*, using the graph_query index if available."""
        if self.graph_query is not None:
            return self.graph_query._functions_for_file(filepath)
        if self.graph is None:
            return []
        functions: list[tuple[int, str]] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "function" or data.get("file") != filepath:
                continue
            functions.append((int(data.get("line_start", 0) or 0), node_id))
        functions.sort()
        return [node_id for _, node_id in functions]

    def _map_diff_to_nodes(
        self,
        diff: str,
        candidate_files: list[str] | None = None,
        changed_ranges: dict[str, list[tuple[int, int]]] | None = None,
    ) -> list[str]:
        if self.graph is None:
            return []
        if changed_ranges is None:
            changed_ranges = self._parse_changed_ranges(diff)
        candidate_paths = list(
            dict.fromkeys([*changed_ranges.keys(), *(candidate_files or [])])
        )
        matched_nodes: list[str] = []
        for filepath in candidate_paths:
            ranges = changed_ranges.get(filepath, [])
            file_matches = []
            if ranges:
                for node_id in self._functions_in_file(filepath):
                    data = self.graph.nodes[node_id]
                    for start, end in ranges:
                        overlaps = not (
                            data.get("line_end", 0) < start or data.get("line_start", 0) > end
                        )
                        if overlaps:
                            file_matches.append(node_id)
                            break
            if file_matches:
                matched_nodes.extend(file_matches)
                continue
            if ranges:
                fallback_matches = self._nearest_file_functions(filepath, ranges)
            else:
                fallback_matches = self._functions_in_file(filepath)[:3]
            if fallback_matches:
                matched_nodes.extend(fallback_matches)
        return list(dict.fromkeys(matched_nodes))

    def _nearest_file_functions(
        self,
        filepath: str,
        ranges: list[tuple[int, int]],
        limit: int = 3,
    ) -> list[str]:
        if self.graph is None:
            return []
        candidates: list[tuple[int, int, str]] = []
        for node_id in self._functions_in_file(filepath):
            data = self.graph.nodes[node_id]
            line_start = int(data.get("line_start", 0) or 0)
            line_end = int(data.get("line_end", 0) or 0)
            distance = min(self._range_distance(line_start, line_end, start, end) for start, end in ranges)
            candidates.append((distance, line_start, node_id))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [node_id for _, _, node_id in candidates[:limit]]

    def _range_distance(
        self,
        node_start: int,
        node_end: int,
        change_start: int,
        change_end: int,
    ) -> int:
        if not (node_end < change_start or node_start > change_end):
            return 0
        if node_end < change_start:
            return change_start - node_end
        return node_start - change_end

    def _parse_changed_ranges(self, diff: str) -> dict[str, list[tuple[int, int]]]:
        ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
        current_file = ""
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("@@") and current_file:
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if not match:
                    continue
                start = int(match.group(1))
                count = int(match.group(2) or "1")
                ranges[current_file].append((start, start + max(count - 1, 0)))
        return ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARES agentic review pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a local repository")
    index_parser.add_argument("--repo", required=True, help="Path to the local repository")

    review_parser = subparsers.add_parser("review", help="Review a GitHub pull request")
    review_parser.add_argument("--repo", required=True, help="GitHub repo in owner/name form")
    review_parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    review_parser.add_argument("--target-dir", default="", help="Local clone directory")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate generated comments against a test set")
    eval_parser.add_argument("--repo", required=True, help="GitHub repo in owner/name form")
    eval_parser.add_argument("--prs", required=True, help="Comma-separated pull request numbers")
    eval_parser.add_argument("--target-dir", default="", help="Local clone directory")

    learn_parser = subparsers.add_parser("learn", help="Collect PR feedback and update the review strategy")
    learn_parser.add_argument("--repo", required=True, help="Path to the local repository clone with .ares feedback state")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AresConfig.from_env()
    pipeline = Pipeline(config)
    if args.command == "index":
        pipeline.index_repo(args.repo)
        print("Graph indexed successfully.")
        return 0
    if args.command == "review":
        comments = pipeline.review_pr(args.repo, args.pr, target_dir=args.target_dir or None)
        print(json.dumps(comments, indent=2))
        return 0
    if args.command == "learn":
        print(json.dumps(pipeline.learn(args.repo), indent=2))
        return 0
    evaluator = Evaluator(pipeline, api_key=config.llm_api_key, model=config.model, provider=config.provider)
    pr_numbers = [int(part.strip()) for part in args.prs.split(",") if part.strip()]
    test_prs = [{"repo_name": args.repo, "pr_number": pr_number, "target_dir": args.target_dir or None} for pr_number in pr_numbers]
    print(json.dumps(evaluator.evaluate(test_prs), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
