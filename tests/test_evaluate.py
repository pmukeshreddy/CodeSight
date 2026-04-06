from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from ares.evaluate import Evaluator
from ares.integrations.github_client import GitHubClient


def test_github_client_labels_review_comments_from_follow_up_commits():
    client = GitHubClient.__new__(GitHubClient)
    review_comments = [
        SimpleNamespace(
            id=1,
            path="app.py",
            body="Guard the payload before calling normalize.",
            created_at=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            original_line=10,
            original_start_line=None,
            line=10,
            start_line=None,
            in_reply_to_id=None,
            user=SimpleNamespace(type="User", login="alice"),
            html_url="https://example.com/comments/1",
        ),
        SimpleNamespace(
            id=2,
            path="app.py",
            body="Nit: reply in thread",
            created_at=datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
            original_line=10,
            original_start_line=None,
            line=10,
            start_line=None,
            in_reply_to_id=1,
            user=SimpleNamespace(type="User", login="alice"),
            html_url="https://example.com/comments/2",
        ),
        SimpleNamespace(
            id=3,
            path="app.py",
            body="Please cover the empty-string case in a test.",
            created_at=datetime(2024, 1, 4, 9, 0, tzinfo=timezone.utc),
            original_line=25,
            original_start_line=None,
            line=25,
            start_line=None,
            in_reply_to_id=None,
            user=SimpleNamespace(type="User", login="bob"),
            html_url="https://example.com/comments/3",
        ),
        SimpleNamespace(
            id=4,
            path="app.py",
            body="Automated reminder",
            created_at=datetime(2024, 1, 4, 9, 5, tzinfo=timezone.utc),
            original_line=30,
            original_start_line=None,
            line=30,
            start_line=None,
            in_reply_to_id=None,
            user=SimpleNamespace(type="Bot", login="ci[bot]"),
            html_url="https://example.com/comments/4",
        ),
    ]
    commits = [
        {
            "committed_at": datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
            "files": {"app.py": [((1, 3), (1, 3))]},
        },
        {
            "committed_at": datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
            "files": {"app.py": [((9, 11), (9, 12))]},
        },
    ]

    labeled = client._label_review_comments(review_comments, commits)

    assert len(labeled) == 2
    assert labeled[0]["comment_id"] == 1
    assert labeled[0]["addressed"] is True
    assert labeled[1]["comment_id"] == 3
    assert labeled[1]["addressed"] is False


def test_github_client_labels_graphql_review_threads():
    client = GitHubClient.__new__(GitHubClient)
    threads = [
        {
            "id": "thread-1",
            "isResolved": True,
            "isOutdated": False,
            "path": "app.py",
            "line": 12,
            "originalLine": 10,
            "startLine": None,
            "originalStartLine": None,
            "comments": {
                "nodes": [
                    {
                        "id": "comment-1",
                        "body": "Guard the payload before calling normalize.",
                        "createdAt": "2024-01-02T10:00:00Z",
                        "author": {"login": "alice", "__typename": "User"},
                        "replyTo": None,
                        "url": "https://example.com/comments/1",
                    },
                    {
                        "id": "comment-2",
                        "body": "Follow-up reply.",
                        "createdAt": "2024-01-02T10:05:00Z",
                        "author": {"login": "alice", "__typename": "User"},
                        "replyTo": {"id": "comment-1"},
                        "url": "https://example.com/comments/2",
                    },
                ]
            },
        },
        {
            "id": "thread-2",
            "isResolved": False,
            "isOutdated": True,
            "path": "app.py",
            "line": None,
            "originalLine": None,
            "startLine": None,
            "originalStartLine": None,
            "comments": {"nodes": []},
        },
    ]

    labeled = client._label_review_threads(threads)

    assert len(labeled) == 1
    assert labeled[0]["comment_id"] == "comment-1"
    assert labeled[0]["addressed"] is True
    assert labeled[0]["file"] == "app.py"
    assert labeled[0]["line_start"] == 10
    assert labeled[0]["line_end"] == 10


def test_github_client_prefers_graphql_review_threads_over_rest_comments():
    client = GitHubClient.__new__(GitHubClient)
    graphql_calls: list[tuple[str, str, int]] = []

    def fake_threads(owner: str, name: str, pr_number: int) -> list[dict]:
        graphql_calls.append((owner, name, pr_number))
        return [
            {
                "id": "thread-1",
                "isResolved": True,
                "isOutdated": False,
                "path": "app.py",
                "line": 12,
                "originalLine": 10,
                "startLine": None,
                "originalStartLine": None,
                "comments": {
                    "nodes": [
                        {
                            "id": "comment-1",
                            "body": "Guard the payload before calling normalize.",
                            "createdAt": "2024-01-02T10:00:00Z",
                            "author": {"login": "alice", "__typename": "User"},
                            "replyTo": None,
                            "url": "https://example.com/comments/1",
                        }
                    ]
                },
            }
        ]

    class FakeRepo:
        def get_pull(self, pr_number: int):
            raise AssertionError("REST fallback should not run when GraphQL threads are available")

    class FakeGitHub:
        def __init__(self) -> None:
            self.requester = object()

        def get_repo(self, repo_name: str):
            assert repo_name == "owner/repo"
            return FakeRepo()

    client.gh = FakeGitHub()
    client._load_review_threads = fake_threads

    labeled = client.get_review_ground_truth("owner/repo", 42)

    assert graphql_calls == [("owner", "repo", 42)]
    assert len(labeled) == 1
    assert labeled[0]["comment_id"] == "comment-1"


def test_github_client_lists_recent_merged_prs_from_graphql():
    client = GitHubClient.__new__(GitHubClient)

    class FakeRequester:
        def graphql_query(self, query: str, variables: dict) -> tuple[dict, dict]:
            assert variables["owner"] == "owner"
            assert variables["name"] == "repo"
            return (
                {},
                {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "nodes": [
                                    {
                                        "number": 42,
                                        "mergedAt": "2024-01-01T00:00:00Z",
                                        "reviewThreads": {"totalCount": 3},
                                    },
                                    {
                                        "number": 41,
                                        "mergedAt": "2023-12-31T00:00:00Z",
                                        "reviewThreads": {"totalCount": 0},
                                    },
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                },
            )

    class FakeGitHub:
        def __init__(self) -> None:
            self.requester = FakeRequester()

    client.gh = FakeGitHub()

    prs = client.list_recent_merged_prs("owner/repo", 10)

    assert prs == [
        {"number": 42, "review_thread_count": 3},
        {"number": 41, "review_thread_count": 0},
    ]


def test_evaluator_fetches_human_comments_when_missing():
    class FakeGitHub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def get_review_ground_truth(self, repo_name: str, pr_number: int) -> list[dict]:
            self.calls.append((repo_name, pr_number))
            return [
                {
                    "file": "app.py",
                    "line_start": 10,
                    "line_end": 10,
                    "comment": "Guard the payload before calling normalize.",
                    "addressed": True,
                }
            ]

    class FakePipeline:
        def __init__(self) -> None:
            self.github = FakeGitHub()

        def review_pr(self, repo_name: str, pr_number: int, target_dir: str | None = None) -> list[dict]:
            raise AssertionError("review_pr should not run when our_comments are provided")

    pipeline = FakePipeline()
    evaluator = Evaluator(pipeline, api_key="")

    result = evaluator.evaluate(
        [
            {
                "repo_name": "owner/repo",
                "pr_number": 42,
                "our_comments": [
                    {
                        "file": "app.py",
                        "line_start": 10,
                        "line_end": 10,
                        "comment": "Guard the payload before calling normalize.",
                        "suggested_code": "payload = payload or ''",
                    }
                ],
            }
        ]
    )

    assert pipeline.github.calls == [("owner/repo", 42)]
    assert result["address_rate"] == 1.0
    assert result["precision"] == 1.0
    assert result["verified_rate"] == 1.0


def test_evaluator_does_not_match_same_line_with_weak_text_similarity():
    evaluator = Evaluator(pipeline=object(), api_key="")

    result = evaluator.evaluate_single(
        {
            "repo_name": "owner/repo",
            "pr_number": 7,
            "human_comments": [
                {
                    "file": "app.py",
                    "line_start": 10,
                    "line_end": 10,
                    "comment": "Add a regression test for empty payload handling.",
                    "addressed": True,
                }
            ],
        },
        [
            {
                "file": "utils.py",
                "line_start": 50,
                "line_end": 50,
                "comment": "This branch can return None and break the caller.",
            }
        ],
    )

    assert result["precision"] == 0.0
    assert result["address_rate"] == 0.0
