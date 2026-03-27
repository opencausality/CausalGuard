"""Graph visualiser for CausalGuard.

Renders the causal failure graph as an interactive HTML file using pyvis.

Colour scheme:
- HAZARD nodes:  orange  (#FF8C00)
- FAILURE nodes: yellow  (#FFD700)
- HARM nodes:    red     (#DC143C)
- Covered edges (mitigation exists): green (#228B22)
- Uncovered edges (gap):            red, bold (#DC143C)

Mitigation names are shown as labels on covered edges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import networkx as nx

from causalguard.data.schema import CoverageResult, Mitigation

logger = logging.getLogger("causalguard.graph.visualizer")

# ── Colour constants ──────────────────────────────────────────────────────────

NODE_COLORS = {
    "HAZARD": "#FF8C00",   # Dark orange
    "FAILURE": "#FFD700",  # Gold/yellow
    "HARM": "#DC143C",     # Crimson red
}
NODE_BORDER_COLORS = {
    "HAZARD": "#CC7000",
    "FAILURE": "#CCA800",
    "HARM": "#AA0020",
}
EDGE_COLOR_COVERED = "#228B22"    # Forest green
EDGE_COLOR_UNCOVERED = "#DC143C"  # Crimson red


def render_graph(
    graph: nx.DiGraph,
    output_path: Path | str,
    coverage_results: Optional[list[CoverageResult]] = None,
    mitigations: Optional[list[Mitigation]] = None,
    title: str = "CausalGuard — Failure Mode Graph",
) -> Path:
    """Render the failure graph as an interactive HTML file.

    Parameters
    ----------
    graph:
        The NetworkX failure graph to render.
    output_path:
        Where to write the HTML file.
    coverage_results:
        Optional path coverage results used to colour edges as covered/uncovered.
    mitigations:
        Optional mitigations used to label covered edges.
    title:
        Title shown in the HTML page.

    Returns
    -------
    Path
        Absolute path to the generated HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("pyvis is required for graph visualisation: pip install pyvis") from exc

    output_path = Path(output_path).resolve()

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.set_options(_pyvis_options(title))

    # ── Compute covered / uncovered edges ─────────────────────────────────
    covered_edges: set[tuple[str, str]] = set()
    edge_mitigation_labels: dict[tuple[str, str], list[str]] = {}

    if coverage_results and mitigations:
        # Build a lookup: which mitigation names cover which failure mode names
        mitigation_by_mode: dict[str, list[str]] = {}
        for m in mitigations:
            for mode_name in m.blocks_failure_modes:
                mitigation_by_mode.setdefault(mode_name, []).append(m.name)

        # For each covered path, mark its edges as covered
        for result in coverage_results:
            if result.is_covered:
                for i in range(len(result.path) - 1):
                    edge = (result.path[i], result.path[i + 1])
                    covered_edges.add(edge)
                    # Collect mitigation labels for this edge
                    node_a, node_b = edge
                    labels: list[str] = []
                    labels.extend(mitigation_by_mode.get(node_a, []))
                    labels.extend(mitigation_by_mode.get(node_b, []))
                    if labels:
                        existing = edge_mitigation_labels.get(edge, [])
                        edge_mitigation_labels[edge] = list(set(existing + labels))

    # ── Add nodes ─────────────────────────────────────────────────────────
    for node, data in graph.nodes(data=True):
        node_type = data.get("node_type", "FAILURE")
        severity = data.get("severity", "HIGH")
        description = data.get("description", "")
        fm_id = data.get("fm_id", "")

        color = NODE_COLORS.get(node_type, NODE_COLORS["FAILURE"])
        border_color = NODE_BORDER_COLORS.get(node_type, NODE_BORDER_COLORS["FAILURE"])

        # Scale node size by severity
        size_map = {"CRITICAL": 35, "HIGH": 28, "MEDIUM": 22, "LOW": 18}
        size = size_map.get(severity, 25)

        tooltip = (
            f"<b>{node}</b><br>"
            f"Type: {node_type}<br>"
            f"Severity: {severity}<br>"
            f"ID: {fm_id}<br><br>"
            f"{description[:200]}{'...' if len(description) > 200 else ''}"
        )

        net.add_node(
            node,
            label=node,
            title=tooltip,
            color={"background": color, "border": border_color, "highlight": {"background": color}},
            size=size,
            font={"size": 13, "color": "#ffffff", "bold": node_type in ("HAZARD", "HARM")},
            shape="ellipse" if node_type == "FAILURE" else "box",
        )

    # ── Add edges ─────────────────────────────────────────────────────────
    for u, v, data in graph.edges(data=True):
        edge_key = (u, v)
        is_covered = edge_key in covered_edges
        mechanism = data.get("mechanism", f"{u} → {v}")
        severity = data.get("severity", "HIGH")

        if is_covered:
            color = EDGE_COLOR_COVERED
            width = 2
            mit_labels = edge_mitigation_labels.get(edge_key, [])
            label = " | ".join(mit_labels[:2]) if mit_labels else ""
            dashes = False
        else:
            color = EDGE_COLOR_UNCOVERED
            width = 3
            label = "GAP"
            dashes = True  # Dashed line for uncovered edges

        tooltip = (
            f"<b>{u} → {v}</b><br>"
            f"Mechanism: {mechanism}<br>"
            f"Severity: {severity}<br>"
            f"Covered: {'Yes' if is_covered else 'NO - COVERAGE GAP'}"
        )

        if coverage_results is not None:
            net.add_edge(
                u,
                v,
                title=tooltip,
                label=label,
                color=color,
                width=width,
                dashes=dashes,
                arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
                font={"size": 11, "color": "#cccccc"},
            )
        else:
            net.add_edge(
                u,
                v,
                title=tooltip,
                color="#888888",
                width=1.5,
                arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
            )

    net.write_html(str(output_path))
    logger.info("Graph rendered to %s (%d nodes, %d edges).",
                output_path, graph.number_of_nodes(), graph.number_of_edges())
    return output_path


def _pyvis_options(title: str) -> str:
    """Return the pyvis configuration JSON string."""
    return """{
  "configure": {"enabled": false},
  "nodes": {
    "borderWidth": 2,
    "shadow": {"enabled": true, "size": 10},
    "font": {"size": 13}
  },
  "edges": {
    "smooth": {"type": "dynamic"},
    "shadow": {"enabled": false},
    "font": {"align": "middle"}
  },
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 150,
      "springConstant": 0.04,
      "damping": 0.09
    },
    "stabilization": {"iterations": 150}
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true
  }
}"""
