from __future__ import annotations

from ares.graph.parser import RepoParser


def test_repo_parser_extracts_nodes_and_edges(sample_repo):
    graph = RepoParser(str(sample_repo)).parse_repo()

    assert "app.py" in graph
    assert "helpers.py" in graph
    assert "app.py::Service.validate" in graph
    assert "app.py::helper" in graph
    assert "helpers.py::normalize" in graph
    assert graph.nodes["app.py::Service.validate"]["params"] == ["self", "payload"]

    assert graph.has_edge("app.py", "helpers.py")
    assert graph.edges["app.py", "helpers.py"]["type"] == "imports"

    assert graph.has_edge("app.py::Service.validate", "helpers.py::normalize")
    call_edge = graph.edges["app.py::Service.validate", "helpers.py::normalize"]
    assert "calls" in call_edge.get("types", [call_edge.get("type")])

    assert graph.has_edge("tests/test_app.py::test_helper", "app.py::helper")
    test_edge = graph.edges["tests/test_app.py::test_helper", "app.py::helper"]
    assert "tests" in test_edge.get("types", [])
