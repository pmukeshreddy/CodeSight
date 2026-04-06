from __future__ import annotations

from ares.feedback import ReviewStrategy
from ares.graph.classifier import NodeClassifier
from ares.graph.parser import RepoParser


def test_classifier_applies_risk_rules(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    classifier = NodeClassifier(graph, str(sample_repo))
    classifier.classify_all()

    assert graph.nodes["tests/test_app.py"]["risk"] == "test"
    assert graph.nodes["app.py::Service.validate"]["risk"] == "critical"
    assert graph.nodes["helpers.py::normalize"]["risk"] == "utility"


def test_classifier_uses_strategy_thresholds(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    classifier = NodeClassifier(
        graph,
        str(sample_repo),
        strategy=ReviewStrategy(bug_fix_freq_critical=2, change_freq_utility=0),
    )
    helper = graph.nodes["app.py::helper"]
    helper["bug_fix_freq"] = 2
    helper["change_freq"] = 1
    graph.nodes["app.py"]["risk"] = "standard"

    assert classifier._classify_function("app.py::helper", helper) == "critical"

    helper["bug_fix_freq"] = 0
    assert classifier._classify_function("app.py::helper", helper) == "standard"


def test_classifier_enriches_only_review_neighborhood(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()
    classifier = NodeClassifier(graph, str(sample_repo))
    classifier.classify_all()

    classifier._count_bug_fix_commits = lambda *args: 2
    classifier._count_total_commits = lambda *args: 4
    classifier._get_last_author = lambda *args: "alice"
    classifier._get_co_changed_files = lambda *args: {}

    assert graph.nodes["app.py::Service.validate"].get("git_metadata_loaded") is None
    assert graph.nodes["app.py::helper"].get("git_metadata_loaded") is None

    classifier.enrich_nodes(["app.py::Service.validate"])

    assert graph.nodes["app.py::Service.validate"]["git_metadata_loaded"] is True
    assert graph.nodes["helpers.py::normalize"]["git_metadata_loaded"] is True
    assert graph.nodes["app.py::helper"].get("git_metadata_loaded") is None
