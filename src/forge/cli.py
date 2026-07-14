"""Forge CLI: doctor, index, plan, run, history."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge import __version__
from forge.config import ForgeSettings
from forge.llm.ollama import OllamaClient
from forge.memory.store import MemoryStore
from forge.orchestrator.loop import ExecutionLoop
from forge.repo.scanner import RepoScanner
from forge.telemetry import Recorder

app = typer.Typer(
    name="forge",
    help="Forge - a local autonomous AI software engineer powered by Ollama.",
    no_args_is_help=True,
)
console = Console()


def _settings(model: Optional[str] = None) -> ForgeSettings:
    settings = ForgeSettings()
    if model:
        settings.model = model
    return settings


def _client(settings: ForgeSettings) -> OllamaClient:
    return OllamaClient(
        host=settings.ollama_host,
        model=settings.model,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        timeout_s=settings.request_timeout_s,
    )


@app.command()
def version() -> None:
    """Print the Forge version."""
    console.print(f"forge {__version__}")


@app.command()
def doctor(model: Optional[str] = typer.Option(None, help="Model to check for.")) -> None:
    """Check that Ollama is reachable and the configured model is available."""
    settings = _settings(model)
    client = _client(settings)
    if not client.ping():
        console.print(f"[red][FAIL] Ollama is not reachable at {settings.ollama_host}[/red]")
        console.print("  Start it with: [bold]ollama serve[/bold]")
        raise typer.Exit(1)
    console.print(f"[green][OK] Ollama reachable at {settings.ollama_host}[/green]")
    models = client.list_models()
    if settings.model in models:
        console.print(f"[green][OK] Model {settings.model} is available[/green]")
    else:
        console.print(
            f"[red][FAIL] Model {settings.model} not found.[/red] Available: {', '.join(models)}"
        )
        console.print(f"  Pull it with: [bold]ollama pull {settings.model}[/bold]")
        raise typer.Exit(1)


@app.command()
def index(
    repo: Path = typer.Option(Path("."), help="Repository to scan."),
) -> None:
    """Scan a repository and print its structure, languages and symbols."""
    workspace = repo.resolve()
    snapshot = RepoScanner(
        workspace, cache_path=workspace / ".forge" / "repo_index.json"
    ).scan()
    console.print(snapshot.summary())


@app.command()
def plan(
    request: str = typer.Argument(..., help="What you want done."),
    repo: Path = typer.Option(Path("."), help="Target repository."),
    model: Optional[str] = typer.Option(None, help="Override the Ollama model."),
) -> None:
    """Generate an execution plan without changing any files."""
    settings = _settings(model)
    workspace = repo.resolve()
    store = MemoryStore(workspace)
    run_id = store.start_run(f"[plan] {request}")
    recorder = Recorder(run_id, workspace, store=store, console=console, verbose=False)

    from forge.agents.planner import Planner

    snapshot = RepoScanner(
        workspace,
        settings.max_tree_entries,
        cache_path=workspace / ".forge" / "repo_index.json",
    ).scan()
    planner = Planner(_client(settings), recorder, settings.max_plan_tasks)
    result = planner.plan(request, snapshot.summary(settings.max_summary_chars))
    store.finish_run(run_id, "success", result.model_dump())

    console.print(Panel(result.summary, title="Plan"))
    table = Table("#", "Task", "Complexity", "Target files")
    for task in result.tasks:
        table.add_row(str(task.id), task.title, task.complexity, "\n".join(task.target_files))
    console.print(table)


@app.command()
def run(
    request: str = typer.Argument(..., help="What you want done."),
    repo: Path = typer.Option(Path("."), help="Target repository."),
    model: Optional[str] = typer.Option(None, help="Override the Ollama model."),
    check: list[str] = typer.Option(
        [], help="Command(s) run after coding and shown to the reviewer, e.g. 'pytest -q'."
    ),
    verbose: bool = typer.Option(True, help="Stream agent activity to the console."),
) -> None:
    """Run the full autonomous loop: plan, code, check, review, iterate."""
    settings = _settings(model)
    workspace = repo.resolve()
    store = MemoryStore(workspace)
    run_id = store.start_run(request)
    recorder = Recorder(run_id, workspace, store=store, console=console, verbose=verbose)
    console.print(
        Panel(f"[bold]{request}[/bold]\nrepo: {workspace}\nrun: {run_id}", title="Forge run")
    )

    loop = ExecutionLoop(
        workspace=workspace,
        llm=_client(settings),
        settings=settings,
        recorder=recorder,
        store=store,
        run_id=run_id,
        check_commands=list(check),
    )
    report = loop.run(request)
    store.finish_run(run_id, report.status, report.model_dump())

    color = {"success": "green", "partial": "yellow"}.get(report.status, "red")
    console.print(
        Panel(f"[{color}]{report.status.upper()}[/{color}]  ({report.duration_s}s)", title="Result")
    )
    if report.error:
        console.print(f"[red]Error:[/red] {report.error}")
    for result in report.task_results:
        mark = "[green]ok[/green]" if result.status == "approved" else "[red]x[/red]"
        console.print(
            f" {mark} task {result.task_id}: {result.title} "
            f"({result.status}, {result.attempts} attempt(s))"
        )
        if result.review and result.review.summary:
            console.print(f"   review: {result.review.summary}")
    if report.changed_files:
        console.print("\n[bold]Changed files[/bold] (originals backed up in .forge/backups/):")
        for path in report.changed_files:
            console.print(f"  {path}")
    if report.diff:
        console.print(Panel(report.diff[:6000], title="Diff"))
    console.print(
        f"\ntokens: {report.usage.prompt_tokens} in / {report.usage.completion_tokens} out"
        f" | report: .forge/runs/{run_id}.json"
    )
    if report.status != "success":
        raise typer.Exit(1)


@app.command()
def history(
    repo: Path = typer.Option(Path("."), help="Target repository."),
    limit: int = typer.Option(10, help="Number of runs to show."),
) -> None:
    """Show recent Forge runs for a repository."""
    store = MemoryStore(repo.resolve())
    runs = store.recent_runs(limit)
    if not runs:
        console.print("No runs recorded yet.")
        return
    table = Table("run id", "status", "started", "task")
    for entry in runs:
        table.add_row(entry["id"], entry["status"], entry["started_at"][:19], entry["task"][:60])
    console.print(table)


if __name__ == "__main__":
    app()
