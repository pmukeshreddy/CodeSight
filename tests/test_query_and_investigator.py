from __future__ import annotations

from ares.config import AresConfig
from ares.agents.investigator import Investigator
from ares.graph.classifier import NodeClassifier
from ares.graph.parser import RepoParser
from ares.graph.query import GraphQuery
from ares.pipeline import Pipeline


def test_graph_query_and_investigator_build_focused_context(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    NodeClassifier(graph, str(sample_repo)).classify_all()
    query = GraphQuery(graph)
    targets = query.get_review_targets(["app.py::Service.validate"])

    assert len(targets) == 1
    context = targets[0]["context"]
    assert context["callers"] == []
    assert context["callees"][0]["node_id"] == "helpers.py::normalize"

    diff = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "--- a/app.py",
            "+++ b/app.py",
            "@@ -4,2 +4,3 @@",
            " class Service:",
            "-    def validate(self, payload):",
            "+    def validate(self, payload):",
            "+        payload = payload or ''",
            "         return normalize(payload)",
        ]
    )
    investigator = Investigator(query, str(sample_repo))
    investigated = investigator.investigate(
        diff,
        "Refactor auth validation flow with no behavior change expected",
        ["app.py::Service.validate"],
        changed_files=["app.py"],
        reviewable_changed_files=["app.py"],
    )

    assert len(investigated) == 1
    assert "Flag behavior changes only" in investigated[0]["review_instruction"]
    assert "@@ -4,2 +4,3 @@" in investigated[0]["diff_hunk"]
    assert "helpers.py::normalize" in investigated[0]["context"]["callees"]


def test_investigator_does_not_treat_versioned_code_pr_as_chore(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    NodeClassifier(graph, str(sample_repo)).classify_all()
    investigator = Investigator(GraphQuery(graph), str(sample_repo))
    diff = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "--- a/app.py",
            "+++ b/app.py",
            "@@ -4,2 +4,3 @@",
            " class Service:",
            "-    def validate(self, payload):",
            "+    def validate(self, payload):",
            "+        payload = payload or ''",
            "         return normalize(payload)",
        ]
    )

    investigated = investigator.investigate(
        diff,
        "Add feature for API versioning support",
        ["app.py::Service.validate"],
        changed_files=["app.py"],
        reviewable_changed_files=["app.py"],
    )

    assert len(investigated) == 1
    assert investigated[0]["pr_intent"].startswith("feature")


def test_investigator_skips_maintenance_only_non_source_pr(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    NodeClassifier(graph, str(sample_repo)).classify_all()
    investigator = Investigator(GraphQuery(graph), str(sample_repo))

    investigated = investigator.investigate(
        "",
        "Dependabot chore: bump dependency versions",
        [],
        changed_files=["poetry.lock"],
        reviewable_changed_files=[],
    )

    assert investigated == []


def test_pipeline_maps_diff_to_nearest_function_when_hunk_hits_module_level(sample_repo):
    pipeline = Pipeline(AresConfig(repo_path=str(sample_repo), workspace_root=str(sample_repo)))
    pipeline.index_repo(str(sample_repo))

    diff = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "--- a/app.py",
            "+++ b/app.py",
            "@@ -1,1 +1,2 @@",
            " from helpers import normalize",
            "+from typing import Any",
        ]
    )

    changed_nodes = pipeline._map_diff_to_nodes(diff)

    assert "app.py::Service.validate" in changed_nodes


def test_pipeline_maps_changed_reviewable_file_without_hunks_to_functions(sample_repo):
    pipeline = Pipeline(AresConfig(repo_path=str(sample_repo), workspace_root=str(sample_repo)))
    pipeline.index_repo(str(sample_repo))

    changed_nodes = pipeline._map_diff_to_nodes("", ["app.py"])

    assert "app.py::Service.validate" in changed_nodes
    assert "app.py::helper" in changed_nodes


def test_investigator_uses_fallback_targets_when_graph_query_filters_everything(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    NodeClassifier(graph, str(sample_repo)).classify_all()
    query = GraphQuery(graph)
    investigator = Investigator(query, str(sample_repo))

    diff = "\n".join(
        [
            "diff --git a/helpers.py b/helpers.py",
            "--- a/helpers.py",
            "+++ b/helpers.py",
            "@@ -1,4 +1,4 @@",
            " def normalize(value):",
            "     if value is None:",
            "         return ''",
            "-    return value.strip()",
            "+    return value.lower().strip()",
        ]
    )

    investigated = investigator.investigate(
        diff,
        "Refactor helper normalization path",
        ["helpers.py::normalize"],
        changed_files=["helpers.py"],
        reviewable_changed_files=["helpers.py"],
    )

    assert len(investigated) == 1
    assert investigated[0]["original_risk"] == "utility"


def test_investigator_uses_file_fallback_targets_when_changed_nodes_is_empty(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    NodeClassifier(graph, str(sample_repo)).classify_all()
    query = GraphQuery(graph)
    investigator = Investigator(query, str(sample_repo))

    investigated = investigator.investigate(
        "",
        "Add feature for API versioning support",
        [],
        changed_files=["app.py"],
        reviewable_changed_files=["app.py"],
    )

    assert len(investigated) >= 1
    assert investigated[0]["file"] == "app.py"
    assert investigated[0]["function_signature"].startswith("def ") or "validate" in investigated[0]["function_signature"]
