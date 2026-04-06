from __future__ import annotations

import networkx as nx

from ares.feedback import ReviewStrategy
from ares.graph.query import GraphQuery


def test_graph_query_skips_low_blast_radius_utility_even_for_logic_changes():
    graph = _utility_graph(
        caller_count=1,
        utility_source="def normalize(value):\n    return value.lower().strip()\n",
    )
    query = GraphQuery(graph)

    targets = query.get_review_targets(
        ["helpers.py::normalize"],
        diff_hunks={
            "helpers.py": [
                {
                    "start": 1,
                    "end": 2,
                    "lines": [
                        "@@ -1,2 +1,2 @@",
                        " def normalize(value):",
                        "-    return value.strip()",
                        "+    return value.lower().strip()",
                    ],
                }
            ]
        },
    )

    assert targets == []


def test_graph_query_promotes_high_blast_radius_utility_with_logic_change():
    graph = _utility_graph(
        caller_count=6,
        utility_source="def normalize(value):\n    return value.lower().strip()\n",
    )
    query = GraphQuery(graph)

    targets = query.get_review_targets(
        ["helpers.py::normalize"],
        diff_hunks={
            "helpers.py": [
                {
                    "start": 1,
                    "end": 2,
                    "lines": [
                        "@@ -1,2 +1,2 @@",
                        " def normalize(value):",
                        "-    return value.strip()",
                        "+    return value.lower().strip()",
                    ],
                }
            ]
        },
    )

    assert len(targets) == 1
    assert targets[0]["risk"] == "standard"
    assert targets[0]["original_risk"] == "utility"
    assert targets[0]["direct_caller_count"] == 6
    assert targets[0]["caller_count"] == 6
    assert targets[0]["change_type"] == "logic"


def test_graph_query_skips_high_blast_radius_utility_for_cosmetic_change():
    graph = _utility_graph(caller_count=6)
    query = GraphQuery(graph)

    targets = query.get_review_targets(
        ["helpers.py::normalize"],
        diff_hunks={
            "helpers.py": [
                {
                    "start": 1,
                    "end": 2,
                    "lines": [
                        "@@ -1,2 +1,2 @@",
                        " def normalize(value):",
                        "-    return( value.strip() )",
                        "+    return value.strip()",
                    ],
                }
            ]
        },
    )

    assert targets == []


def test_graph_query_uses_strategy_blast_radius_threshold():
    graph = _utility_graph(
        caller_count=6,
        utility_source="def normalize(value):\n    return value.lower().strip()\n",
    )
    query = GraphQuery(graph, strategy=ReviewStrategy(utility_blast_radius_threshold=7))

    targets = query.get_review_targets(
        ["helpers.py::normalize"],
        diff_hunks={
            "helpers.py": [
                {
                    "start": 1,
                    "end": 2,
                    "lines": [
                        "@@ -1,2 +1,2 @@",
                        " def normalize(value):",
                        "-    return value.strip()",
                        "+    return value.lower().strip()",
                    ],
                }
            ]
        },
    )

    assert targets == []


def test_graph_query_uses_transitive_blast_radius_for_utility_promotion():
    graph = _transitive_utility_graph(
        direct_caller_count=2,
        transitive_leaf_count=12,
        utility_source="def normalize(value):\n    return value.lower().strip()\n",
    )
    query = GraphQuery(graph, strategy=ReviewStrategy(utility_blast_radius_threshold=5))

    targets = query.get_review_targets(
        ["helpers.py::normalize"],
        diff_hunks={
            "helpers.py": [
                {
                    "start": 1,
                    "end": 2,
                    "lines": [
                        "@@ -1,2 +1,2 @@",
                        " def normalize(value):",
                        "-    return value.strip()",
                        "+    return value.lower().strip()",
                    ],
                }
            ]
        },
    )

    assert len(targets) == 1
    assert targets[0]["direct_caller_count"] == 2
    assert targets[0]["caller_count"] == 14


def test_graph_query_can_build_fallback_targets_for_changed_utility():
    graph = _utility_graph(
        caller_count=1,
        utility_source="def normalize(value):\n    return value.lower().strip()\n",
    )
    query = GraphQuery(graph)

    targets = query.get_fallback_targets(
        ["helpers.py::normalize"],
        diff_hunks={
            "helpers.py": [
                {
                    "start": 1,
                    "end": 2,
                    "lines": [
                        "@@ -1,2 +1,2 @@",
                        " def normalize(value):",
                        "-    return value.strip()",
                        "+    return value.lower().strip()",
                    ],
                }
            ]
        },
    )

    assert len(targets) == 1
    assert targets[0]["fallback_target"] is True
    assert targets[0]["original_risk"] == "utility"


def test_graph_query_can_build_file_fallback_targets():
    graph = _utility_graph(caller_count=1)
    graph.add_node(
        "helpers.py::extra",
        type="function",
        risk="standard",
        source="def extra(value):\n    return value\n",
        file="helpers.py",
        line_start=10,
        line_end=11,
        signature="def extra(value)",
        bug_fix_freq=0,
        change_freq=0,
    )
    query = GraphQuery(graph)

    targets = query.get_file_fallback_targets(["helpers.py"], limit_per_file=2)

    assert len(targets) == 2
    assert all(target["fallback_target"] for target in targets)


def _utility_graph(caller_count: int, utility_source: str = "def normalize(value):\n    return value.strip()\n") -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        "helpers.py::normalize",
        type="function",
        risk="utility",
        source=utility_source,
        file="helpers.py",
        line_start=1,
        line_end=2,
        signature="def normalize(value)",
        bug_fix_freq=0,
        change_freq=0,
    )
    for index in range(caller_count):
        caller_id = f"app.py::caller_{index}"
        graph.add_node(
            caller_id,
            type="function",
            risk="standard",
            source=f"def caller_{index}(value):\n    return normalize(value)\n",
            file="app.py",
            line_start=10 + (index * 2),
            line_end=11 + (index * 2),
            signature=f"def caller_{index}(value)",
        )
        graph.add_edge(caller_id, "helpers.py::normalize", type="calls")
    return graph


def _transitive_utility_graph(
    direct_caller_count: int,
    transitive_leaf_count: int,
    utility_source: str = "def normalize(value):\n    return value.strip()\n",
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        "helpers.py::normalize",
        type="function",
        risk="utility",
        source=utility_source,
        file="helpers.py",
        line_start=1,
        line_end=2,
        signature="def normalize(value)",
        bug_fix_freq=0,
        change_freq=0,
    )
    wrappers = []
    for index in range(direct_caller_count):
        wrapper_id = f"service.py::wrapper_{index}"
        wrappers.append(wrapper_id)
        graph.add_node(
            wrapper_id,
            type="function",
            risk="standard",
            source=f"def wrapper_{index}(value):\n    return normalize(value)\n",
            file="service.py",
            line_start=20 + (index * 2),
            line_end=21 + (index * 2),
            signature=f"def wrapper_{index}(value)",
        )
        graph.add_edge(wrapper_id, "helpers.py::normalize", type="calls")
    for index in range(transitive_leaf_count):
        leaf_id = f"api.py::entry_{index}"
        wrapper_id = wrappers[index % len(wrappers)]
        graph.add_node(
            leaf_id,
            type="function",
            risk="standard",
            source=f"def entry_{index}(value):\n    return wrapper_{index % len(wrappers)}(value)\n",
            file="api.py",
            line_start=100 + (index * 2),
            line_end=101 + (index * 2),
            signature=f"def entry_{index}(value)",
        )
        graph.add_edge(leaf_id, wrapper_id, type="calls")
    return graph
