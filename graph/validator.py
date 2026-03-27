"""Graph validator for CausalGuard.

Validates structural properties required for meaningful safety case analysis:
- Graph must have at least one node.
- Graph must have at least one hazard node (root trigger).
- Graph must have at least one harm node (terminal impact).
- Graph must have at least one path from a hazard to a harm.
- Graph must be a DAG (no cycles — enforced by builder, re-checked here).
"""

from __future__ import annotations

import logging

import networkx as nx

from causalguard.exceptions import GraphBuildError

logger = logging.getLogger("causalguard.graph.validator")


def validate_graph(
    graph: nx.DiGraph,
    hazards: list[str],
    harms: list[str],
) -> None:
    """Validate that the failure graph is structurally sound for safety analysis.

    Parameters
    ----------
    graph:
        The constructed failure graph.
    hazards:
        List of identified hazard node names.
    harms:
        List of identified harm node names.

    Raises
    ------
    GraphBuildError
        If any structural validation constraint is violated.
    """
    # 1. Graph must have nodes
    if graph.number_of_nodes() == 0:
        raise GraphBuildError(
            "Failure graph is empty — no nodes were added. "
            "Ensure failure modes were extracted successfully."
        )

    # 2. Must be a DAG
    if not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        raise GraphBuildError(
            f"Failure graph contains cycles (not a DAG): {cycles}. "
            "This is a causal modelling error — failure propagation cannot be circular."
        )

    # 3. Must have at least one hazard
    if not hazards:
        raise GraphBuildError(
            "No hazard nodes found in the failure graph. "
            "At least one HAZARD node (root trigger condition) is required. "
            f"Current nodes: {list(graph.nodes)[:10]}"
        )

    # 4. Must have at least one harm
    if not harms:
        raise GraphBuildError(
            "No harm nodes found in the failure graph. "
            "At least one HARM node (terminal impact) is required. "
            f"Current nodes: {list(graph.nodes)[:10]}"
        )

    # 5. Must have at least one reachable path from a hazard to a harm
    has_path = False
    for hazard in hazards:
        for harm in harms:
            if hazard != harm and nx.has_path(graph, hazard, harm):
                has_path = True
                break
        if has_path:
            break

    if not has_path:
        raise GraphBuildError(
            f"No path exists from any hazard to any harm in the failure graph. "
            f"Hazards: {hazards}. Harms: {harms}. "
            "Ensure downstream_effects relationships connect hazards to harms."
        )

    logger.info(
        "Graph validation passed: %d nodes, %d edges, %d hazards, %d harms.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(hazards),
        len(harms),
    )


def graph_summary(
    graph: nx.DiGraph,
    hazards: list[str],
    harms: list[str],
) -> dict[str, object]:
    """Return a summary statistics dictionary about the graph.

    Parameters
    ----------
    graph:
        The failure graph.
    hazards:
        Identified hazard nodes.
    harms:
        Identified harm nodes.

    Returns
    -------
    dict
        Summary statistics including node/edge counts and path count.
    """
    path_count = 0
    for hazard in hazards:
        for harm in harms:
            if hazard != harm:
                try:
                    paths = list(nx.all_simple_paths(graph, hazard, harm))
                    path_count += len(paths)
                except (nx.NetworkXError, nx.NodeNotFound):
                    pass

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "hazard_count": len(hazards),
        "harm_count": len(harms),
        "total_paths": path_count,
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "is_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
    }
