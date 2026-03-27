"""Causal failure graph builder.

Converts extracted :class:`~causalguard.data.schema.FailureMode` objects
into a directed acyclic NetworkX graph where:

- Nodes represent failure modes, typed as HAZARD / FAILURE / HARM.
- Edges represent ``downstream_effects`` relationships between failure modes.
- Self-loops and cycles are detected and removed with warnings.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from causalguard.data.schema import FailureMode, SafetyEdge
from causalguard.exceptions import GraphBuildError

logger = logging.getLogger("causalguard.graph.builder")


def build_failure_graph(
    failure_modes: list[FailureMode],
) -> tuple[nx.DiGraph, list[SafetyEdge]]:
    """Build a causal DAG of failure modes from their downstream_effects links.

    Nodes are annotated with:
    - ``node_type``: HAZARD / FAILURE / HARM
    - ``severity``: CRITICAL / HIGH / MEDIUM / LOW
    - ``description``: description text
    - ``id``: short identifier

    Edges are annotated with:
    - ``mechanism``: description of how cause leads to effect
    - ``severity``: inherited from the causing node

    Parameters
    ----------
    failure_modes:
        List of extracted failure modes.

    Returns
    -------
    tuple[nx.DiGraph, list[SafetyEdge]]
        The constructed directed graph and the list of safety edges.

    Raises
    ------
    GraphBuildError
        If no failure modes are provided, or if the resulting graph contains
        cycles that cannot be resolved.
    """
    if not failure_modes:
        raise GraphBuildError("Cannot build a failure graph with no failure modes.")

    # Build a name → FailureMode index for fast lookup
    by_name: dict[str, FailureMode] = {fm.name: fm for fm in failure_modes}

    graph = nx.DiGraph()

    # Add all nodes first
    for fm in failure_modes:
        graph.add_node(
            fm.name,
            node_type=fm.node_type,
            severity=fm.severity,
            description=fm.description,
            fm_id=fm.id,
        )
        logger.debug("Added node: %s [%s, %s]", fm.name, fm.node_type, fm.severity)

    # Add edges from downstream_effects
    safety_edges: list[SafetyEdge] = []

    for fm in failure_modes:
        for effect_name in fm.downstream_effects:
            if effect_name not in by_name:
                logger.warning(
                    "Failure mode '%s' lists downstream effect '%s' which is not "
                    "in the failure modes list. Adding as an unknown HARM node.",
                    fm.name,
                    effect_name,
                )
                # Add the unknown effect as a HARM node so the graph remains connected
                graph.add_node(
                    effect_name,
                    node_type="HARM",
                    severity="HIGH",
                    description=f"Unknown harm: {effect_name}",
                    fm_id="UNKNOWN",
                )

            if fm.name == effect_name:
                logger.warning("Self-loop detected on '%s' — skipping.", fm.name)
                continue

            graph.add_edge(
                fm.name,
                effect_name,
                mechanism=f"{fm.name} leads to {effect_name}",
                severity=fm.severity,
            )
            safety_edges.append(
                SafetyEdge(
                    cause=fm.name,
                    effect=effect_name,
                    mechanism=f"{fm.name} leads to {effect_name}",
                    severity=fm.severity,
                )
            )
            logger.debug("Added edge: %s → %s", fm.name, effect_name)

    # Cycle detection — remove edges that form cycles with a warning
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            logger.warning(
                "Cycle(s) detected in failure graph: %s. "
                "Removing last edge in each cycle to enforce DAG property.",
                cycles,
            )
            for cycle in cycles:
                # Remove the last edge in the cycle to break it
                if len(cycle) >= 2:
                    u, v = cycle[-1], cycle[0]
                    if graph.has_edge(u, v):
                        graph.remove_edge(u, v)
                        safety_edges = [
                            e for e in safety_edges if not (e.cause == u and e.effect == v)
                        ]
                        logger.warning("Removed cycle-breaking edge: %s → %s", u, v)
    except nx.NetworkXError as exc:
        raise GraphBuildError(f"Cycle detection failed: {exc}") from exc

    logger.info(
        "Built failure graph: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph, safety_edges


def identify_hazards(
    graph: nx.DiGraph,
    failure_modes: list[FailureMode],
) -> list[str]:
    """Find HAZARD nodes: nodes explicitly typed HAZARD or with in-degree 0.

    Parameters
    ----------
    graph:
        The constructed failure graph.
    failure_modes:
        The full list of failure modes used to build the graph.

    Returns
    -------
    list[str]
        Node names that are hazards.
    """
    # Nodes explicitly marked as HAZARD
    explicit_hazards = {
        fm.name for fm in failure_modes if fm.node_type == "HAZARD"
    }

    # Nodes with in-degree 0 in the graph (root nodes)
    root_nodes = {node for node in graph.nodes if graph.in_degree(node) == 0}

    # Union of both sets, filtered to nodes actually in the graph
    hazards = list((explicit_hazards | root_nodes) & set(graph.nodes))

    if not hazards:
        logger.warning(
            "No hazard nodes found. Root nodes: %s, Explicit hazards: %s.",
            root_nodes,
            explicit_hazards,
        )

    logger.debug("Identified hazard nodes: %s", hazards)
    return sorted(hazards)


def identify_harms(
    graph: nx.DiGraph,
    failure_modes: Optional[list[FailureMode]] = None,
) -> list[str]:
    """Find HARM nodes: nodes explicitly typed HARM or with out-degree 0.

    Parameters
    ----------
    graph:
        The constructed failure graph.
    failure_modes:
        Optional list of failure modes to check for explicit HARM typing.

    Returns
    -------
    list[str]
        Node names that are harms.
    """
    # Nodes with out-degree 0 (terminal leaf nodes)
    leaf_nodes = {node for node in graph.nodes if graph.out_degree(node) == 0}

    # Nodes explicitly typed as HARM
    explicit_harms: set[str] = set()
    if failure_modes:
        explicit_harms = {fm.name for fm in failure_modes if fm.node_type == "HARM"}

    harms = list((explicit_harms | leaf_nodes) & set(graph.nodes))

    if not harms:
        logger.warning(
            "No harm nodes found. Leaf nodes: %s, Explicit harms: %s.",
            leaf_nodes,
            explicit_harms,
        )

    logger.debug("Identified harm nodes: %s", harms)
    return sorted(harms)
