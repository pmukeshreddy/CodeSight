from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from github import Github
except ImportError:  # pragma: no cover - optional dependency
    Github = None


class GitHubClient:
    def __init__(self, token: str):
        if Github is None:
            raise RuntimeError("PyGithub is required for GitHub integration.")
        self.gh = Github(token)

    def get_pr_data(self, repo_name: str, pr_number: int) -> dict:
        repo = self.gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        files = list(pr.get_files())
        commits = list(pr.get_commits())
        return {
            "title": pr.title or "",
            "description": pr.body or "",
            "commit_messages": [commit.commit.message for commit in commits],
            "changed_files": [changed.filename for changed in files],
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "head_ref": f"pull/{pr_number}/head",
            "head_sha": pr.head.sha,
            "diff": self._build_unified_diff(files),
        }

    def get_pr_overview(self, repo_name: str, pr_number: int) -> dict:
        repo = self.gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        return {
            "pr_number": pr_number,
            "title": pr.title or "",
            "description": pr.body or "",
            "changed_files": [changed.filename for changed in pr.get_files()],
        }

    def get_review_ground_truth(self, repo_name: str, pr_number: int) -> list[dict]:
        owner, name = repo_name.split("/", 1)
        threads = self._load_review_threads(owner, name, pr_number)
        if threads:
            return self._label_review_threads(threads)
        repo = self.gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        commits = self._load_pr_commits(repo, pr)
        review_comments = list(pr.get_review_comments())
        return self._label_review_comments(review_comments, commits)

    def list_recent_merged_prs(
        self,
        repo_name: str,
        limit: int,
    ) -> list[dict]:
        owner, name = repo_name.split("/", 1)
        requester = getattr(self.gh, "requester", None)
        if requester is None:
            return []
        query = """
query RecentMergedPullRequests($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 100, after: $cursor, states: [MERGED], orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        mergedAt
        reviewThreads {
          totalCount
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
""".strip()
        items: list[dict] = []
        cursor = None
        while len(items) < limit:
            try:
                _, data = requester.graphql_query(
                    query,
                    {
                        "owner": owner,
                        "name": name,
                        "cursor": cursor,
                    },
                )
            except Exception:
                return items
            pull_requests = (
                data.get("data", {})
                .get("repository", {})
                .get("pullRequests", {})
            )
            nodes = pull_requests.get("nodes", []) or []
            if not nodes:
                break
            for node in nodes:
                if not isinstance(node, dict) or not node.get("mergedAt"):
                    continue
                items.append(
                    {
                        "number": int(node.get("number", 0) or 0),
                        "review_thread_count": int(
                            (node.get("reviewThreads") or {}).get("totalCount", 0) or 0
                        ),
                    }
                )
                if len(items) >= limit:
                    break
            page_info = pull_requests.get("pageInfo", {}) or {}
            if len(items) >= limit or not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor:
                break
        return [item for item in items if item.get("number")]

    def get_pending_comment_feedback(
        self,
        repo_name: str,
        pr_number: int,
        pending_comments: list[dict],
    ) -> list[dict]:
        repo = self.gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        commits = self._load_pr_commits(repo, pr)
        merged = bool(getattr(pr, "merged", False))
        state = getattr(pr, "state", "open") or "open"
        feedback = []
        for comment in pending_comments:
            created_at = self._parse_datetime(comment.get("posted_at", ""))
            addressed = any(
                self._commit_touches_lines(
                    commit,
                    comment.get("file", ""),
                    int(comment.get("line_start", 0) or 0),
                    int(comment.get("line_end", 0) or 0),
                )
                for commit in commits
                if created_at is None or (
                    commit.get("committed_at") is not None and commit["committed_at"] > created_at
                )
            )
            outcome = "pending"
            label = ""
            if addressed:
                outcome = "addressed"
                label = "upvote"
            elif merged or state != "open":
                outcome = "ignored"
                label = "downvote"
            feedback.append(
                {
                    **comment,
                    "merged": merged,
                    "pr_state": state,
                    "outcome": outcome,
                    "feedback_label": label,
                }
            )
        return feedback

    def post_review_comments(self, repo_name: str, pr_number: int, comments: list[dict]) -> None:
        if not comments:
            return
        repo = self.gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        review_comments = [
            {
                "path": comment["file"],
                "line": comment["line_end"],
                "side": "RIGHT",
                "body": comment.get("github_body") or self._format_comment(comment),
            }
            for comment in comments
        ]
        pr.create_review(
            body=f"ARES Review: {len(review_comments)} issues found",
            comments=review_comments,
            event="COMMENT",
        )

    def clone_repo(self, repo_name: str, branch: str, target_dir: str, depth: int = 1) -> None:
        target = Path(target_dir)
        clone_url = f"https://github.com/{repo_name}.git"
        depth_args = ["--depth", str(max(1, depth))]
        if target.exists() and (target / ".git").exists():
            self._run_git(["git", "-C", str(target), "fetch", "origin", branch, *depth_args])
            self._run_git(["git", "-C", str(target), "checkout", "-B", branch, "FETCH_HEAD"])
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        self._run_git(["git", "clone", *depth_args, "--branch", branch, clone_url, str(target)])

    def clone_repo_ref(self, repo_name: str, ref: str, target_dir: str, checkout_name: str = "ares-target", depth: int = 1) -> None:
        target = Path(target_dir)
        clone_url = f"https://github.com/{repo_name}.git"
        depth_args = ["--depth", str(max(1, depth))]
        if target.exists() and (target / ".git").exists():
            self._run_git(["git", "-C", str(target), "fetch", "origin", ref, *depth_args])
            self._run_git(["git", "-C", str(target), "checkout", "-B", checkout_name, "FETCH_HEAD"])
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        self._run_git(["git", "clone", *depth_args, "--no-checkout", clone_url, str(target)])
        self._run_git(["git", "-C", str(target), "fetch", "origin", ref, *depth_args])
        self._run_git(["git", "-C", str(target), "checkout", "-B", checkout_name, "FETCH_HEAD"])

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess:
        verbose = os.getenv("ARES_GIT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
        if verbose:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
        if result.returncode != 0:
            cmd_str = " ".join(args[:3]) + " ..."
            stderr_excerpt = (result.stderr or "").strip()[:200]
            raise RuntimeError(
                f"Git command failed (exit {result.returncode}): {cmd_str}"
                + (f"\n{stderr_excerpt}" if stderr_excerpt else "")
            )

    def _build_unified_diff(self, files) -> str:
        chunks = []
        for file in files:
            patch = file.patch or ""
            chunks.append(
                "\n".join(
                    [
                        f"diff --git a/{file.filename} b/{file.filename}",
                        f"--- a/{file.filename}",
                        f"+++ b/{file.filename}",
                        patch,
                    ]
                )
            )
        return "\n".join(chunks)

    def _format_comment(self, comment: dict) -> str:
        body = f"**{comment.get('severity', 'warning').upper()}**: {comment.get('comment', '')}"
        suggested = comment.get("suggested_code", "").rstrip()
        if suggested:
            body += f"\n\n```suggestion\n{suggested}\n```"
        return body

    def _load_pr_commits(self, repo, pr) -> list[dict]:
        commits: list[dict] = []
        for pull_commit in pr.get_commits():
            commit = repo.get_commit(pull_commit.sha)
            touched_files = {}
            for changed_file in getattr(commit, "files", []) or []:
                filename = getattr(changed_file, "filename", "")
                if filename:
                    touched_files[filename] = self._parse_patch_ranges(getattr(changed_file, "patch", "") or "")
            commits.append(
                {
                    "sha": commit.sha,
                    "committed_at": self._normalize_datetime(
                        getattr(getattr(commit.commit, "committer", None), "date", None)
                        or getattr(getattr(commit.commit, "author", None), "date", None)
                    ),
                    "files": touched_files,
                }
            )
        commits.sort(key=lambda item: item["committed_at"] or datetime.min.replace(tzinfo=timezone.utc))
        return commits

    def _label_review_comments(self, review_comments, commits: list[dict]) -> list[dict]:
        labeled_comments = []
        for comment in review_comments:
            if self._should_skip_review_comment(comment):
                continue
            line_start, line_end = self._extract_line_span(comment)
            if line_start is None or line_end is None:
                continue
            created_at = self._normalize_datetime(getattr(comment, "created_at", None))
            filepath = getattr(comment, "path", "")
            addressed = any(
                self._commit_touches_lines(commit, filepath, line_start, line_end)
                for commit in commits
                if created_at is None or (
                    commit.get("committed_at") is not None and commit["committed_at"] > created_at
                )
            )
            labeled_comments.append(
                {
                    "comment_id": getattr(comment, "id", 0),
                    "file": filepath,
                    "line_start": line_start,
                    "line_end": line_end,
                    "comment": getattr(comment, "body", ""),
                    "addressed": addressed,
                    "author": getattr(getattr(comment, "user", None), "login", ""),
                    "created_at": created_at.isoformat() if created_at is not None else "",
                    "url": getattr(comment, "html_url", ""),
                }
            )
        return labeled_comments

    def _load_review_threads(self, owner: str, name: str, pr_number: int) -> list[dict]:
        requester = getattr(self.gh, "requester", None)
        if requester is None:
            return []
        query = """
query ReviewThreads($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          originalLine
          startLine
          originalStartLine
          comments(first: 20) {
            nodes {
              id
              body
              createdAt
              author {
                __typename
                login
              }
              replyTo {
                id
              }
              url
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
""".strip()
        threads: list[dict] = []
        cursor = None
        while True:
            try:
                _, data = requester.graphql_query(
                    query,
                    {
                        "owner": owner,
                        "name": name,
                        "number": pr_number,
                        "cursor": cursor,
                    },
                )
            except Exception:
                return []
            review_threads = (
                data.get("data", {})
                .get("repository", {})
                .get("pullRequest", {})
                .get("reviewThreads", {})
            )
            nodes = review_threads.get("nodes", []) or []
            threads.extend(node for node in nodes if isinstance(node, dict))
            page_info = review_threads.get("pageInfo", {}) or {}
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor:
                break
        return threads

    def _label_review_threads(self, threads: list[dict]) -> list[dict]:
        labeled_threads = []
        for thread in threads:
            filepath = str(thread.get("path") or "")
            if not filepath:
                continue
            line_start, line_end = self._thread_line_span(thread)
            if line_start is None or line_end is None:
                continue
            comments = (
                thread.get("comments", {}).get("nodes", [])
                if isinstance(thread.get("comments"), dict)
                else []
            )
            comment = self._select_thread_comment(comments)
            if comment is None:
                continue
            author = comment.get("author") or {}
            login = str(author.get("login") or "")
            author_type = str(author.get("__typename") or "")
            if self._is_bot_identity(login, author_type):
                continue
            body = str(comment.get("body") or "").strip()
            if not body:
                continue
            labeled_threads.append(
                {
                    "comment_id": comment.get("id") or thread.get("id"),
                    "file": filepath,
                    "line_start": line_start,
                    "line_end": line_end,
                    "comment": body,
                    "addressed": bool(thread.get("isResolved") or thread.get("isOutdated")),
                    "author": login,
                    "created_at": str(comment.get("createdAt") or ""),
                    "url": comment.get("url", ""),
                    "source": "graphql_review_thread",
                }
            )
        return labeled_threads

    def _should_skip_review_comment(self, comment) -> bool:
        if not getattr(comment, "body", "").strip():
            return True
        if not getattr(comment, "path", ""):
            return True
        if getattr(comment, "in_reply_to_id", None) is not None:
            return True
        user = getattr(comment, "user", None)
        if user is None:
            return False
        user_type = (getattr(user, "type", "") or "").lower()
        login = (getattr(user, "login", "") or "").lower()
        return self._is_bot_identity(login, user_type)

    def _is_bot_identity(self, login: str, user_type: str) -> bool:
        return (user_type or "").lower() == "bot" or (login or "").lower().endswith("[bot]")

    def _thread_line_span(self, thread: dict) -> tuple[int | None, int | None]:
        line_start = self._first_line_number(
            thread.get("originalStartLine"),
            thread.get("startLine"),
            thread.get("originalLine"),
            thread.get("line"),
        )
        line_end = self._first_line_number(
            thread.get("originalLine"),
            thread.get("line"),
            line_start,
        )
        if line_start is None and line_end is None:
            return None, None
        if line_start is None:
            line_start = line_end
        if line_end is None:
            line_end = line_start
        return (line_start, line_end) if line_start <= line_end else (line_end, line_start)

    def _select_thread_comment(self, comments: list[dict]) -> dict | None:
        selected = None
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            body = str(comment.get("body") or "").strip()
            if not body:
                continue
            author = comment.get("author") or {}
            if self._is_bot_identity(str(author.get("login") or ""), str(author.get("__typename") or "")):
                continue
            reply_to = comment.get("replyTo")
            if reply_to in (None, {}):
                return comment
            if selected is None:
                selected = comment
        return selected

    def _extract_line_span(self, comment) -> tuple[int | None, int | None]:
        line_start = self._first_line_number(
            getattr(comment, "original_start_line", None),
            getattr(comment, "start_line", None),
            getattr(comment, "original_line", None),
            getattr(comment, "line", None),
        )
        line_end = self._first_line_number(
            getattr(comment, "original_line", None),
            getattr(comment, "line", None),
            line_start,
        )
        if line_start is None and line_end is None:
            return None, None
        if line_start is None:
            line_start = line_end
        if line_end is None:
            line_end = line_start
        return (line_start, line_end) if line_start <= line_end else (line_end, line_start)

    def _first_line_number(self, *candidates) -> int | None:
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _parse_patch_ranges(self, patch: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        ranges = []
        for line in patch.splitlines():
            match = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not match:
                continue
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            ranges.append(
                (
                    self._line_range(old_start, old_count),
                    self._line_range(new_start, new_count),
                )
            )
        return ranges

    def _line_range(self, start: int, count: int) -> tuple[int, int]:
        return start, start + max(count - 1, 0)

    def _commit_touches_lines(
        self,
        commit: dict,
        filepath: str,
        line_start: int,
        line_end: int,
    ) -> bool:
        for old_range, new_range in commit.get("files", {}).get(filepath, []):
            if self._ranges_overlap((line_start, line_end), old_range):
                return True
            if self._ranges_overlap((line_start, line_end), new_range):
                return True
        return False

    def _ranges_overlap(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        return not (left[1] < right[0] or left[0] > right[1])

    def _normalize_datetime(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return self._normalize_datetime(datetime.fromisoformat(value))
        except ValueError:
            return None
