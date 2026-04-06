from __future__ import annotations

import os

from ares.config import AresConfig
from scripts.eval_comment_sample import (
    collapse_reason,
    effective_eval_max_comments,
    preflight_candidate_prs,
)


def test_collapse_reason_identifies_zero_target_path():
    assert collapse_reason({"target_count": 0}) == "no_targets"


def test_collapse_reason_identifies_zero_survivor_path():
    assert collapse_reason({"target_count": 2, "survivor_count": 0}) == "review_loop_zero_survivors"


def test_collapse_reason_identifies_ranker_and_bundle_paths():
    assert collapse_reason({"target_count": 2, "survivor_count": 1, "ranked_count_before_bundle": 0}) == "ranker_zero_comments"
    assert (
        collapse_reason(
            {
                "target_count": 2,
                "survivor_count": 1,
                "ranked_count_before_bundle": 1,
                "final_comment_count": 0,
                "bundle_drop_count": 1,
            }
        )
        == "bundle_dropped_all"
    )


def test_collapse_reason_identifies_reviewable_files_without_indexed_functions():
    assert (
        collapse_reason(
            {
                "reviewable_changed_file_count": 1,
                "changed_node_count": 0,
                "target_count": 0,
            }
        )
        == "no_indexed_functions_in_reviewable_files"
    )


def test_preflight_candidate_prs_rejects_non_reviewable_and_maintenance_prs():
    class FakeGitHub:
        def get_pr_overview(self, repo_name: str, pr_number: int) -> dict:
            assert repo_name == "owner/repo"
            if pr_number == 1:
                return {
                    "title": "Docs update",
                    "description": "",
                    "changed_files": ["docs/index.md"],
                }
            if pr_number == 2:
                return {
                    "title": "Dependabot chore",
                    "description": "bump dependency versions",
                    "changed_files": ["poetry.lock"],
                }
            return {
                "title": "Fix auth edge case",
                "description": "",
                "changed_files": ["app.py", "tests/test_app.py"],
            }

    class FakePipeline:
        def __init__(self) -> None:
            self.github = FakeGitHub()

    approved, rejected = preflight_candidate_prs(
        FakePipeline(),
        "owner/repo",
        [1, 2, 3],
        {1: 1, 2: 1, 3: 2},
        parallelism=2,
    )

    assert approved == [3]
    assert rejected == {"no_reviewable_source_files": 1, "maintenance_only_pr": 1}


def test_config_can_use_eval_comment_cap_without_changing_default():
    previous = os.environ.get("ARES_MAX_COMMENTS")
    os.environ["ARES_MAX_COMMENTS"] = "10"
    try:
        assert AresConfig.from_env().max_comments == 10
        assert AresConfig().max_comments == 10
    finally:
        if previous is None:
            os.environ.pop("ARES_MAX_COMMENTS", None)
        else:
            os.environ["ARES_MAX_COMMENTS"] = previous

    if previous is None:
        assert AresConfig().max_comments == 3


def test_effective_eval_max_comments_auto_raises_for_large_historical_runs():
    assert effective_eval_max_comments(3, 100) == 10
    assert effective_eval_max_comments(10, 100) == 10
    assert effective_eval_max_comments(3, 25) == 3
