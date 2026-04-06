from __future__ import annotations

from ares.agents._llm import LLMAdapter
from ares.utils.text_similarity import batch_similarities

# Matching weights and thresholds.
# Text uses sentence-transformers cosine similarity (CRScore paper threshold ~0.73 for pure text).
# File/line bonuses lower the effective text threshold when ARES and human comment on the same code.
HUMAN_MATCH_THRESHOLD = 0.45
HUMAN_MATCH_TEXT_WEIGHT = 0.60
HUMAN_MATCH_FILE_BONUS = 0.15
HUMAN_MATCH_LINE_BONUS = 0.25
HUMAN_MATCH_PROXIMITY_BONUS = 0.10  # lines within window but not overlapping
HUMAN_MATCH_PROXIMITY_WINDOW = 15

# A comment is "plausible" (novel valid find) when it sits on a changed line but has no human match.
PLAUSIBLE_DIFF_SENTINEL = "No matching diff hunk found."


class Evaluator:
    def __init__(
        self,
        pipeline,
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
    ):
        self.pipeline = pipeline
        self.client = LLMAdapter(api_key=api_key, model=model, provider=provider)

    def evaluate(self, test_prs: list[dict], use_batch: bool = False) -> dict:
        results = []
        # When use_batch=True, review all PRs via the Batch API (50% cost savings).
        if use_batch and hasattr(self.pipeline, "batch_review_prs"):
            prs_needing_review = [
                (idx, pr_data)
                for idx, pr_data in enumerate(test_prs)
                if pr_data.get("our_comments") is None
            ]
            if prs_needing_review:
                batch_configs = [
                    {
                        "repo_name": pr_data["repo_name"],
                        "pr_number": pr_data["pr_number"],
                        "target_dir": pr_data.get("target_dir"),
                    }
                    for _, pr_data in prs_needing_review
                ]
                batch_results = self.pipeline.batch_review_prs(batch_configs)
                for (idx, _), comments in zip(prs_needing_review, batch_results):
                    test_prs[idx]["our_comments"] = comments
        for pr_data in test_prs:
            repo_name = pr_data["repo_name"]
            pr_number = pr_data["pr_number"]
            enriched_pr_data = self._hydrate_human_comments(pr_data)
            comments = pr_data.get("our_comments")
            if comments is None:
                comments = self.pipeline.review_pr(
                    repo_name,
                    pr_number,
                    target_dir=pr_data.get("target_dir"),
                )
            results.append(self.evaluate_single(enriched_pr_data, comments))
        if not results:
            return {
                "address_rate": 0.0,
                "precision": 0.0,
                "comments_per_pr": 0.0,
                "verified_rate": 0.0,
                "per_pr": [],
            }
        address_rate = sum(item["address_rate"] for item in results) / len(results)
        precision = sum(item["precision"] for item in results) / len(results)
        comments_per_pr = sum(item["comments_generated"] for item in results) / len(results)
        verified_rate = sum(item["verified_rate"] for item in results) / len(results)
        return {
            "address_rate": round(address_rate, 3),
            "precision": round(precision, 3),
            "comments_per_pr": round(comments_per_pr, 3),
            "verified_rate": round(verified_rate, 3),
            "per_pr": results,
        }

    def _hydrate_human_comments(self, pr_data: dict) -> dict:
        if pr_data.get("human_comments") is not None:
            return pr_data
        github = getattr(self.pipeline, "github", None)
        if github is None:
            return {**pr_data, "human_comments": []}
        human_comments = github.get_review_ground_truth(
            pr_data["repo_name"],
            pr_data["pr_number"],
        )
        return {**pr_data, "human_comments": human_comments}

    def evaluate_single(self, pr_data: dict, our_comments: list[dict]) -> dict:
        human_comments = pr_data.get("human_comments", [])
        addressed = 0
        matched = 0
        plausible = 0
        # Separate LLM comments from static findings for accurate metrics.
        # Static findings (ruff/semgrep) rarely match human review comments
        # and dilute the address rate when counted.
        llm_comments = [c for c in our_comments if c.get("source") != "static"]
        static_comments = [c for c in our_comments if c.get("source") == "static"]
        for our_comment in llm_comments:
            best = self._best_human_match(our_comment, human_comments)
            if best is not None:
                matched += 1
                if best.get("addressed", False):
                    addressed += 1
            elif self._is_plausible(our_comment):
                plausible += 1
        n_llm = len(llm_comments)
        n_total = len(our_comments)
        precision = matched / n_llm if n_llm else 0.0
        address_rate = addressed / n_llm if n_llm else 0.0
        plausible_rate = plausible / n_llm if n_llm else 0.0
        verified_rate = (
            sum(1 for c in our_comments if c.get("suggested_code")) / n_total if n_total else 0.0
        )
        return {
            "repo_name": pr_data.get("repo_name", ""),
            "pr_number": pr_data.get("pr_number", 0),
            "address_rate": address_rate,
            "precision": precision,
            "plausible_rate": plausible_rate,
            "comments_generated": n_llm,
            "static_comments": len(static_comments),
            "verified_rate": verified_rate,
        }

    def _is_plausible(self, comment: dict) -> bool:
        """True when the comment sits on a changed diff line (novel valid find, not a hallucination)."""
        diff_hunk = comment.get("diff_hunk", "") or ""
        return bool(diff_hunk.strip()) and diff_hunk.strip() != PLAUSIBLE_DIFF_SENTINEL

    def _best_human_match(self, our_comment: dict, human_comments: list[dict]) -> dict | None:
        if not human_comments:
            return None
        our_text = our_comment.get("comment", "")
        our_start = our_comment.get("line_start", 0) or 0
        our_end = our_comment.get("line_end", 0) or our_start
        our_file = our_comment.get("file", "")
        human_texts = [h.get("comment", "") for h in human_comments]
        text_scores = batch_similarities(our_text, human_texts)
        best = None
        best_score = 0.0
        for idx, human in enumerate(human_comments):
            same_file = our_file == human.get("file", "")
            h_start = human.get("line_start", 0) or 0
            h_end = human.get("line_end", 0) or h_start
            overlap = not (our_end < h_start or our_start > h_end)
            proximity = (
                not overlap
                and min(abs(our_start - h_end), abs(our_end - h_start)) <= HUMAN_MATCH_PROXIMITY_WINDOW
            )
            score = (
                (text_scores[idx] * HUMAN_MATCH_TEXT_WEIGHT)
                + (HUMAN_MATCH_FILE_BONUS if same_file else 0.0)
                + (HUMAN_MATCH_LINE_BONUS if overlap else 0.0)
                + (HUMAN_MATCH_PROXIMITY_BONUS if proximity else 0.0)
            )
            if score > best_score:
                best = human
                best_score = score
        return best if best_score >= HUMAN_MATCH_THRESHOLD else None
