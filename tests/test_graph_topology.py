"""Standalone tests for compiled-graph topology serialization.

Run with:

    PYTHONPATH=src python3 tests/test_graph_topology.py

The end-to-end ``WorkflowGraph.topology()`` path needs langgraph installed (it
compiles a real StateGraph), so it is skipped when langgraph is absent. The pure
``serialize_graph_topology`` projection is always exercised against a stub graph.
"""

import ast
import os
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_passed = 0


def check(name, cond):
    global _passed
    assert cond, f"FAILED: {name}"
    _passed += 1
    print(f"  ok - {name}")


class _Edge:
    def __init__(self, source, target, data=None, conditional=False):
        self.source = source
        self.target = target
        self.data = data
        self.conditional = conditional


class _Drawable:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


class _Compiled:
    def __init__(self, drawable):
        self._drawable = drawable

    def get_graph(self):
        return self._drawable


def _load_serializer():
    """Load just ``serialize_graph_topology`` without importing heavy deps.

    The module imports langgraph/mongodb at top level, so we extract the single
    pure function via AST instead of importing the whole module locally.
    """
    src_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "application",
        "langgraph",
        "models",
        "langraph_model.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    ns = {"Any": typing.Any, "Optional": typing.Optional, "Dict": dict}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "serialize_graph_topology":
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, "<serializer>", "exec"), ns)
            return ns["serialize_graph_topology"]
    raise AssertionError("serialize_graph_topology not found")


def test_serialize_graph_topology():
    serialize = _load_serializer()
    drawable = _Drawable(
        nodes={
            "__start__": 1,
            "generative": 1,
            "execute_script_tool": 1,
            "__end__": 1,
        },
        edges=[
            _Edge("__start__", "generative"),
            _Edge("generative", "execute_script_tool", data="script", conditional=True),
            _Edge("generative", "__end__", data="end", conditional=True),
            _Edge("execute_script_tool", "generative"),
        ],
    )
    topo = serialize(_Compiled(drawable))

    check("entry resolved from __start__ edge", topo["entry"] == "generative")
    check("start/end terminals preserved", topo["start"] == "__start__" and topo["end"] == "__end__")
    check("all nodes serialized", len(topo["nodes"]) == 4)

    labels = {(e["source"], e["target"]): e["label"] for e in topo["edges"]}
    check("conditional branch key becomes label", labels[("generative", "execute_script_tool")] == "script")
    check("plain edge has no label", labels[("execute_script_tool", "generative")] is None)
    conds = {(e["source"], e["target"]): e["conditional"] for e in topo["edges"]}
    check("conditional flag carried", conds[("generative", "__end__")] is True)


def test_workflow_graph_topology_end_to_end():
    try:
        from application.langgraph.models.langraph_model import (
            DEFAULT_GRAPH_KEY,
            topology_for_graph,
        )
    except Exception as exc:  # langgraph / deps absent locally → skip
        print(f"  skip - end-to-end topology (deps unavailable: {exc.__class__.__name__})")
        return

    topo = topology_for_graph()
    ids = {n["id"] for n in topo["nodes"]}
    check("compiled graph has generative node", "generative" in ids)
    check("compiled graph has start/end", "__start__" in ids and "__end__" in ids)
    check("graph key attached", topo.get("graph") == DEFAULT_GRAPH_KEY)
    check("entry is generative", topo["entry"] == "generative")


def main():
    print("test_graph_topology:")
    test_serialize_graph_topology()
    test_workflow_graph_topology_end_to_end()
    print(f"\nAll graph-topology checks passed: {_passed}")


if __name__ == "__main__":
    main()
