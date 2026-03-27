"""CausalGuard CLI — Typer-based command-line interface.

Commands:
    build     Extract failure modes and build a complete safety case
    verify    Verify coverage of an existing safety case
    gaps      Show only uncovered (gap) paths with recommendations
    show      Visualize the safety case causal DAG
    providers List LLM provider status
    serve     Run the optional REST API server
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from causalguard import __version__

app = typer.Typer(
    name="causalguard",
    help="Build causal safety cases for AI systems.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger("causalguard.cli")


# ── Callbacks ─────────────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]CausalGuard[/] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """CausalGuard — build causal safety cases for AI systems."""
    from causalguard.config import Settings, configure_logging

    settings = Settings(log_level="DEBUG" if verbose else "INFO")
    configure_logging(settings)


# ── build ──────────────────────────────────────────────────────────────────────


@app.command()
def build(
    system: Path = typer.Option(
        ...,
        "--system",
        "-s",
        help="Path to the system description text file.",
        exists=True,
    ),
    incidents: Path = typer.Option(
        ...,
        "--incidents",
        "-i",
        help="Path to the incident logs text file.",
        exists=True,
    ),
    mitigations: Path = typer.Option(
        ...,
        "--mitigations",
        "-m",
        help="Path to the mitigations text file.",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save the SafetyCase JSON to this path.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Open an interactive visualisation after building.",
    ),
) -> None:
    """Build a safety case from system documents.

    Runs the full pipeline: LLM extraction → graph construction →
    path coverage verification → gap detection → report.

    Example:

        causalguard build --system system.txt --incidents incidents.txt \\
            --mitigations mitigations.txt --output safety_case.json
    """
    from causalguard.cases.builder import SafetyCaseBuilder
    from causalguard.cases.exporter import format_text_report, save_safety_case
    from causalguard.config import Settings
    from causalguard.exceptions import CausalGuardError
    from causalguard.extraction.failures import FailureModeExtractor
    from causalguard.extraction.mitigations import MitigationExtractor
    from causalguard.llm.adapter import LLMAdapter

    system_text = system.read_text(encoding="utf-8")
    incidents_text = incidents.read_text(encoding="utf-8")
    mitigations_text = mitigations.read_text(encoding="utf-8")

    console.print("[dim]Extracting failure modes via LLM…[/]")

    try:
        settings = Settings()
        adapter = LLMAdapter(settings=settings)

        failure_extractor = FailureModeExtractor(llm=adapter, settings=settings)
        failure_modes = failure_extractor.extract(
            system_description=system_text,
            incident_logs=incidents_text,
        )
        console.print(f"  [green]✓[/] {len(failure_modes)} failure modes extracted")

        mitigation_extractor = MitigationExtractor(llm=adapter, settings=settings)
        extracted_mitigations = mitigation_extractor.extract(
            mitigations_text=mitigations_text,
            failure_modes=failure_modes,
        )
        console.print(f"  [green]✓[/] {len(extracted_mitigations)} mitigations mapped")

        builder = SafetyCaseBuilder(settings=settings)
        case = builder.build(
            system_name=system.stem,
            failure_modes=failure_modes,
            mitigations=extracted_mitigations,
            model_used=settings.litellm_model,
            source_system_description=system_text,
            source_incident_logs=incidents_text,
            source_mitigations_text=mitigations_text,
        )

    except CausalGuardError as exc:
        console.print(f"[red]Build failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n" + format_text_report(case))

    if output:
        save_safety_case(case, output)
        console.print(f"\n[green]✓[/] Safety case saved to [bold]{output}[/]")

    if show:
        from causalguard.graph.visualizer import visualize_safety_case

        vis_path = output.with_suffix(".html") if output else Path("safety_case.html")
        visualize_safety_case(case, output_path=vis_path, show=True)
        console.print(f"[green]✓[/] Graph opened: [bold]{vis_path}[/]")


# ── verify ─────────────────────────────────────────────────────────────────────


@app.command()
def verify(
    case_path: Path = typer.Option(
        ...,
        "--case",
        "-c",
        help="Path to a SafetyCase JSON file.",
        exists=True,
    ),
) -> None:
    """Verify coverage of an existing safety case.

    Loads a saved SafetyCase and prints the full coverage report.

    Example:

        causalguard verify --case safety_case.json
    """
    from causalguard.cases.exporter import format_text_report, load_safety_case
    from causalguard.exceptions import CausalGuardError

    try:
        case = load_safety_case(case_path)
    except CausalGuardError as exc:
        console.print(f"[red]Error loading safety case:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(format_text_report(case))


# ── gaps ───────────────────────────────────────────────────────────────────────


@app.command()
def gaps(
    case_path: Path = typer.Option(
        ...,
        "--case",
        "-c",
        help="Path to a SafetyCase JSON file.",
        exists=True,
    ),
) -> None:
    """Show only uncovered (gap) paths with recommendations.

    Example:

        causalguard gaps --case safety_case.json
    """
    from causalguard.cases.exporter import load_safety_case
    from causalguard.exceptions import CausalGuardError

    try:
        case = load_safety_case(case_path)
    except CausalGuardError as exc:
        console.print(f"[red]Error loading safety case:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not case.uncovered_paths:
        console.print(
            Panel(
                f"[green]No coverage gaps found.[/] All {len(case.coverage_results)} paths are covered.",
                border_style="green",
                title="Coverage Gaps",
            )
        )
        return

    console.print(
        Panel(
            f"[red]{len(case.uncovered_paths)} uncovered path(s)[/] out of "
            f"{len(case.coverage_results)} total paths.\n"
            f"Coverage: {case.coverage_percentage:.1f}%",
            border_style="red",
            title="Coverage Gaps",
        )
    )

    for i, result in enumerate(case.uncovered_paths, 1):
        console.print(f"\n[bold red]Gap {i}:[/] {result.path_string}")
        if result.gap_at_edge:
            console.print(f"  [dim]First uncovered edge:[/] {result.gap_at_edge}")

    if case.recommendations:
        console.print("\n[bold]Recommendations:[/]")
        for j, rec in enumerate(case.recommendations, 1):
            console.print(f"  {j}. {rec}")


# ── show ───────────────────────────────────────────────────────────────────────


@app.command()
def show(
    case_path: Path = typer.Option(
        ...,
        "--case",
        "-c",
        help="Path to a SafetyCase JSON file.",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save the visualisation HTML to this path.",
    ),
) -> None:
    """Visualise the safety case causal DAG in the browser.

    Hazard nodes in orange, harm nodes in red, covered edges in green,
    uncovered edges in bold red.

    Example:

        causalguard show --case safety_case.json
    """
    from causalguard.cases.exporter import load_safety_case
    from causalguard.exceptions import CausalGuardError
    from causalguard.graph.visualizer import visualize_safety_case

    try:
        case = load_safety_case(case_path)
    except CausalGuardError as exc:
        console.print(f"[red]Error loading safety case:[/] {exc}")
        raise typer.Exit(code=1) from exc

    vis_path = output or case_path.with_suffix(".html")
    rendered = visualize_safety_case(case, output_path=vis_path, show=True)
    console.print(f"[green]✓[/] Graph opened: [bold]{rendered}[/]")


# ── providers ──────────────────────────────────────────────────────────────────


@app.command()
def providers() -> None:
    """Check LLM provider status.

    Lists which providers are configured and reachable.
    """
    from causalguard.config import LLMProvider, Settings
    from causalguard.llm.adapter import LLMAdapter

    settings = Settings()
    table = Table(title="LLM Providers", border_style="cyan")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status", justify="center")

    for provider in LLMProvider:
        is_active = provider == settings.llm_provider
        status = "[green]active[/]" if is_active else "[dim]—[/]"
        table.add_row(provider.value, settings.litellm_model if is_active else "—", status)

    console.print(table)

    adapter = LLMAdapter(settings=settings)
    try:
        adapter.health_check()
        console.print(f"\n[green]✓[/] Active provider reachable: {settings.litellm_model}")
    except Exception as exc:
        console.print(f"\n[red]✗[/] Active provider unreachable: {exc}")
        console.print("[dim]Tip: ensure Ollama is running, or set CAUSALGUARD_LLM_PROVIDER.[/]")


# ── serve ───────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Listen address."),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port."),
) -> None:
    """Run the CausalGuard REST API server.

    Example:

        causalguard serve --port 8000
    """
    import uvicorn

    console.print(
        f"[bold cyan]CausalGuard API[/] v{__version__} "
        f"starting on [bold]http://{host}:{port}[/]"
    )
    console.print(f"  [dim]Docs:[/] http://{host}:{port}/docs")
    uvicorn.run("causalguard.api.server:create_app", host=host, port=port, factory=True)
