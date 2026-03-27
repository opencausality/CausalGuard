"""Path coverage verification for CausalGuard.

Core safety question: for every path from a HAZARD node to a HARM node,
is there at least one mitigation that blocks at least one node on that path?

A node is "blocked" by a mitigation if the mitigation lists that node's name
in its ``blocks_failure_modes`` field.

A path is "covered" if at least one mitigation blocks at least one node
in the path (including the hazard and harm endpoints).
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from causalguard.data.schema import CoverageResult, FailureMode, Mitigation
from causalguard.exceptions import VerificationError

logger = logging.getLogger("causalguard.verification.coverage")


def check_path_coverage(
    path: list[str],
    graph: nx.DiGraph,
    mitigations: list[Mitigation],
    failure_modes: list[FailureMode],
) -> CoverageResult:
    """Check if a single hazard→harm path is covered by at least one mitigation.

    A path is covered if at least one mitigation's ``blocks_failure_modes``
    list contains the name of at least one node in the path.

    Parameters
    ----------
    path:
        Ordered list of node names from hazard to harm.
    graph:
        The failure graph (used for edge metadata).
    mitigations:
        All extracted mitigations.
    failure_modes:
        All extracted failure modes (used for node metadata lookup).

    Returns
    -------
    CoverageResult
        Coverage assessment for this path.

    Raises
    ------
    VerificationError
        If path is empty or contains only one node.
    """
    if len(path) < 2:
        raise VerificationError(
            f"Path must have at least 2 nodes (hazard and harm). Got: {path}"
        )

    # Build a lookup of which mitigations cover which node names
    # A mitigation covers a node if the node name is in blocks_failure_modes
    node_to_mitigations: dict[str, list[str]] = {}
    for node in path:
        covering = [m.name for m in mitigations if node in m.blocks_failure_modes]
        node_to_mitigations[node] = covering

    # Collect all mitigations that cover any node in this path
    all_blocking: list[str] = []
    for covering in node_to_mitigations.values():
        all_blocking.extend(covering)
    # Deduplicate while preserving order
    seen: set[str] = set()
    blocking_mitigations: list[str] = []
    for m_name in all_blocking:
        if m_name not in seen:
            seen.add(m_name)
            blocking_mitigations.append(m_name)

    is_covered = len(blocking_mitigations) > 0

    # Find the first edge in the path with no mitigation on either endpoint
    gap_at_edge: Optional[str] = None
    if not is_covered:
        # Path has no coverage at all — report the first unmitigated edge
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            gap_at_edge = f"{u} -> {v}"
            break
    else:
        # Path is covered overall, but check if there's a specific edge with no mitigation
        # (for informational purposes — a covered path might still have weak spots)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            u_covered = bool(node_to_mitigations.get(u))
            v_covered = bool(node_to_mitigations.get(v))
            if not u_covered and not v_covered:
                gap_at_edge = f"{u} -> {v}"
                break

    path_string = " -> ".join(path)
    coverage_gap = not is_covered

    result = CoverageResult(
        path=path,
        path_string=path_string,
        is_covered=is_covered,
        blocking_mitigations=blocking_mitigations,
        coverage_gap=coverage_gap,
        gap_at_edge=gap_at_edge,
    )

    logger.debug(
        "Path %s: covered=%s, blocking=%s",
        path_string,
        is_covered,
        blocking_mitigations,
    )
    return result


def verify_all_paths(
    graph: nx.DiGraph,
    hazards: list[str],
    harms: list[str],
    mitigations: list[Mitigation],
    failure_modes: list[FailureMode],
) -> list[CoverageResult]:
    """Find all hazard→harm paths and verify coverage for each.

    Uses NetworkX ``all_simple_paths`` to enumerate every possible path
    between each hazard-harm pair.

    Parameters
    ----------
    graph:
        The failure graph.
    hazards:
        Names of HAZARD nodes (starting points).
    harms:
        Names of HARM nodes (ending points).
    mitigations:
        All extracted mitigations.
    failure_modes:
        All extracted failure modes.

    Returns
    -------
    list[CoverageResult]
        One ``CoverageResult`` per distinct hazard→harm path.

    Raises
    ------
    VerificationError
        If hazards or harms lists are empty.
    """
    if not hazards:
        raise VerificationError("Cannot verify coverage: no hazard nodes provided.")
    if not harms:
        raise VerificationError("Cannot verify coverage: no harm nodes provided.")

    coverage_results: list[CoverageResult] = []
    total_paths = 0

    for hazard in hazards:
        for harm in harms:
            if hazard == harm:
                continue

            if not nx.has_path(graph, hazard, harm):
                logger.debug("No path from '%s' to '%s'.", hazard, harm)
                continue

            try:
                paths = list(nx.all_simple_paths(graph, hazard, harm))
            except (nx.NetworkXError, nx.NodeNotFound) as exc:
                logger.warning(
                    "Could not enumerate paths from '%s' to '%s': %s",
                    hazard, harm, exc,
                )
                continue

            for path in paths:
                total_paths += 1
                result = check_path_coverage(
                    path=path,
                    graph=graph,
                    mitigations=mitigations,
                    failure_modes=failure_modes,
                )
                coverage_results.append(result)

    logger.info(
        "Verified %d paths across %d hazard-harm pairs. "
        "Covered: %d, Uncovered: %d.",
        total_paths,
        len(hazards) * len(harms),
        sum(1 for r in coverage_results if r.is_covered),
        sum(1 for r in coverage_results if not r.is_covered),
    )
    return coverage_results
