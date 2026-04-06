from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(slots=True)
class AresConfig:
    repo_path: str = ""
    workspace_root: str = field(
        default_factory=lambda: str(Path.cwd() / ".ares-workdir")
    )
    provider: str = field(
        default_factory=lambda: os.getenv("ARES_PROVIDER", "anthropic")
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "ARES_MODEL", "claude-sonnet-4-6"
        )
    )
    llm_api_key: str = field(default_factory=lambda: "")
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    pinecone_api_key: str = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("ARES_PINECONE_INDEX", "ares-comments")
    )
    pinecone_namespace: str = field(
        default_factory=lambda: os.getenv("ARES_PINECONE_NAMESPACE", "default")
    )
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    max_comments: int = field(
        default_factory=lambda: int(os.getenv("ARES_MAX_COMMENTS", "3"))
    )
    max_review_passes: int = field(
        default_factory=lambda: int(os.getenv("ARES_REVIEW_MAX_PASSES", "1"))
    )
    lightweight_model: str = field(
        default_factory=lambda: os.getenv("ARES_LIGHTWEIGHT_MODEL", "claude-haiku-4-5-20251001")
    )
    actionability_filter: bool = field(
        default_factory=lambda: os.getenv("ARES_ACTIONABILITY_FILTER", "1") == "1"
    )
    review_aggregation_runs: int = field(
        default_factory=lambda: int(os.getenv("ARES_REVIEW_AGGREGATION_RUNS", "1"))
    )

    def __post_init__(self) -> None:
        if not self.llm_api_key:
            if self.provider == "openai":
                self.llm_api_key = os.getenv("OPENAI_API_KEY", "")
            else:
                self.llm_api_key = os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def graph_store_dir(self) -> str:
        base = Path(self.repo_path) if self.repo_path else Path(self.workspace_root)
        return str(base / ".ares")

    @classmethod
    def from_env(cls, repo_path: str = "", workspace_root: str | None = None) -> "AresConfig":
        return cls(
            repo_path=repo_path or os.getenv("ARES_REPO_PATH", ""),
            workspace_root=workspace_root or os.getenv("ARES_WORKSPACE_ROOT", str(Path.cwd() / ".ares-workdir")),
        )
