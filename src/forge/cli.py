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
from forge.llm.factory import make_client
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


def _client(settings: ForgeSettings):
    return make_client(settings)


@app.command()
def version() -> None:
    """Print the Forge version."""
    console.print(f"forge {__version__}")


@app.command()
def doctor(model: Optional[str] = typer.Option(None, help="Model to check for.")) -> None:
    """Check that Ollama is reachable and the configured model is available."""
    settings = _settings(model)
    client = _client(settings)
    endpoint = (
        settings.openai_base_url if settings.provider == "openai" else settings.ollama_host
    )
    backend = "LLM server" if settings.provider == "openai" else "Ollama"
    if not client.ping():
        console.print(f"[red][FAIL] {backend} is not reachable at {endpoint}[/red]")
        if settings.provider != "openai":
            console.print("  Start it with: [bold]ollama serve[/bold]")
        raise typer.Exit(1)
    console.print(f"[green][OK] {backend} reachable at {endpoint}[/green]")
    models = client.list_models()
    if settings.model in models:
        console.print(f"[green][OK] Model {settings.model} is available[/green]")
    else:
        console.print(
            f"[red][FAIL] Model {settings.model} not found.[/red] Available: {', '.join(models)}"
        )
        if settings.provider != "openai":
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


@app.command(name="app")
def desktop_app(
    repo: Path = typer.Option(Path("."), help="Initial project folder (changeable in-app)."),
    model: Optional[str] = typer.Option(None, help="Override the Ollama model."),
    port: int = typer.Option(8322, help="Local port for the app backend."),
) -> None:
    """Open Forge as a desktop app: chat window, folder picker, model switcher."""
    import threading

    import uvicorn

    from forge.server.app import create_app

    try:
        import webview
    except ImportError as exc:  # pragma: no cover - packaging problem
        console.print(f"[red]pywebview is not installed:[/red] {exc}")
        console.print("Install it with: pip install pywebview")
        raise typer.Exit(1) from exc

    settings = _settings(model)
    workspace = repo.resolve()
    backend = create_app(workspace, settings)

    server = uvicorn.Server(
        uvicorn.Config(backend, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    class Bridge:
        def pick_folder(self):
            result = window.create_file_dialog(webview.FOLDER_DIALOG)
            if result:
                return result[0] if isinstance(result, (list, tuple)) else result
            return None

    window = webview.create_window(
        "Forge — local AI engineer",
        f"http://127.0.0.1:{port}/app",
        width=1180,
        height=820,
        min_size=(860, 600),
        js_api=Bridge(),
        background_color="#0d1117",
    )
    webview.start()


@app.command()
def shortcut() -> None:
    """Put a Forge shortcut on the Windows desktop (launches the app window)."""
    import subprocess
    import sys

    forge_exe = Path(sys.executable).parent / "forge.exe"
    if not forge_exe.exists():
        console.print(f"[red]forge.exe not found at {forge_exe}[/red]")
        raise typer.Exit(1)
    script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        "$lnk = $ws.CreateShortcut([IO.Path]::Combine("
        "[Environment]::GetFolderPath('Desktop'), 'Forge.lnk')); "
        f"$lnk.TargetPath = '{forge_exe}'; "
        "$lnk.Arguments = 'app'; "
        f"$lnk.WorkingDirectory = '{Path.home()}'; "
        "$lnk.Description = 'Forge - local AI software engineer'; "
        "$lnk.Save()"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode == 0:
        console.print("[green]Created 'Forge' shortcut on your desktop.[/green]")
    else:
        console.print(f"[red]Failed:[/red] {completed.stderr.strip()}")
        raise typer.Exit(1)


@app.command()
def chat(
    repo: Path = typer.Option(Path("."), help="Target repository."),
    model: Optional[str] = typer.Option(None, help="Override the Ollama model."),
    auto: bool = typer.Option(
        False, "--auto", help="Skip permission prompts (auto-approve writes/commands)."
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume the previous chat session."),
) -> None:
    """Interactive session — work with Forge like Claude Code, on your repo."""
    from rich.markdown import Markdown
    from rich.prompt import Confirm

    from forge.chat import commands as chat_commands
    from forge.chat.session import ChatSession
    from forge.safety.permissions import PermissionPolicy

    settings = _settings(model)
    workspace = repo.resolve()

    def approver(tool_name: str, detail: str) -> bool:
        console.print(f"\n[yellow]Forge wants to run[/yellow] [bold]{tool_name}[/bold] {detail}")
        return Confirm.ask("Allow?", default=True)

    policy = PermissionPolicy("auto") if auto else PermissionPolicy("ask", approver)
    store = MemoryStore(workspace)
    recorder = Recorder("chat", workspace, store=store, console=console, verbose=True)
    with console.status("[dim]indexing repository…[/dim]"):
        session = ChatSession(
            workspace, _client(settings), settings, policy=policy, recorder=recorder
        )
    if resume and session.load_transcript():
        console.print(f"[dim]Resumed previous session ({len(session.history)} messages).[/dim]")

    mode = "auto-approve" if auto else "ask before writes/commands"
    console.print(
        Panel(
            f"[bold]Forge chat[/bold] — {settings.model} · {mode}\n"
            f"repo: {workspace}\nType a request, /help for commands, /exit to leave.",
            border_style="dim",
        )
    )
    while True:
        try:
            line = console.input("\n[bold cyan]forge>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye.[/dim]")
            break
        if not line:
            continue
        if chat_commands.is_command(line):
            result = chat_commands.execute(session, line)
            if result.text:
                console.print(result.text)
            if result.should_exit:
                break
            continue
        try:
            reply = session.send(line)
        except Exception as exc:  # noqa: BLE001 — REPL must survive errors
            console.print(f"[red]error:[/red] {exc}")
            continue
        console.print(Markdown(reply))
    store.close()


@app.command()
def serve(
    repo: Path = typer.Option(Path("."), help="Target repository."),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8321, help="Port for the API + dashboard."),
    model: Optional[str] = typer.Option(None, help="Override the Ollama model."),
) -> None:
    """Start the Forge API server and web dashboard."""
    import uvicorn

    from forge.server.app import create_app

    settings = _settings(model)
    workspace = repo.resolve()
    console.print(f"Forge dashboard: [bold]http://{host}:{port}[/bold]  (repo: {workspace})")
    uvicorn.run(create_app(workspace, settings), host=host, port=port, log_level="warning")


@app.command()
def memory(
    repo: Path = typer.Option(Path("."), help="Target repository."),
    limit: int = typer.Option(20, help="Number of lessons to show."),
) -> None:
    """Show lessons Forge has learned from past runs on this repository."""
    store = MemoryStore(repo.resolve())
    lessons = store.lessons(limit)
    if not lessons:
        console.print("No lessons recorded yet.")
        return
    table = Table("when", "task", "status", "issues")
    for lesson in lessons:
        table.add_row(
            lesson["ts"][:19],
            lesson["task_title"][:50],
            lesson["status"],
            "\n".join(lesson["issues"][:3]) or "-",
        )
    console.print(table)


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


@app.command(name="eval")
def evaluate(
    tier: Optional[int] = typer.Option(None, help="Only run tier 1, 2 or 3."),
    task: Optional[str] = typer.Option(None, help="Comma-separated task ids."),
    model: Optional[str] = typer.Option(None, help="Model under test."),
    effort: str = typer.Option("smart", help="fast | smart | genius."),
    ablation: str = typer.Option("all-gates", help="Gate configuration to test."),
    seeds: int = typer.Option(1, help="Repeats per task (run-to-run variance)."),
    out: Optional[Path] = typer.Option(None, help="Write the JSON report here."),
) -> None:
    """Run the SWE-micro benchmark and report success + behavioural metrics."""
    from forge.evals.runner import ABLATIONS, run_suite, write_report

    if ablation not in ABLATIONS:
        console.print(f"[red]Unknown ablation {ablation!r}.[/red] "
                      f"Options: {', '.join(ABLATIONS)}")
        raise typer.Exit(2)

    settings = _settings(model)
    settings.effort = effort
    ids = [t.strip() for t in task.split(",")] if task else None

    console.print(
        Panel(
            f"model [bold]{settings.model}[/bold] · effort [bold]{effort}[/bold] · "
            f"gates [bold]{ablation}[/bold] · seeds [bold]{seeds}[/bold]",
            title="SWE-micro",
        )
    )

    def progress(result) -> None:
        mark = "[green]PASS[/green]" if result.solved else "[red]FAIL[/red]"
        console.print(
            f"  {mark} {result.task_id} (T{result.tier}) "
            f"{result.duration_s}s · {result.metrics.summary_line()}"
        )

    report = run_suite(
        llm_factory=lambda: _client(settings),
        settings=settings,
        tier=tier,
        ids=ids,
        seeds=seeds,
        ablation=ablation,
        on_result=progress,
    )

    summary = report.summary()
    table = Table("metric", "value")
    table.add_row("task success rate", f"{summary['task_success_rate'] * 100:.1f}%")
    table.add_row("solved", f"{summary['solved']}/{summary['total']}")
    for tier_no, stats in summary["by_tier"].items():
        table.add_row(f"  tier {tier_no}", f"{stats['solved']}/{stats['total']}")
    for name, stats in summary["metrics"].items():
        if name == "totals":
            continue
        value = stats["mean"]
        table.add_row(
            name.replace("_", " "),
            "n/a" if value is None else f"{value * 100:.1f}%  (n={stats['n']})",
        )
    totals = summary["metrics"]["totals"]
    table.add_row("tool calls", f"{totals['tool_calls']} ({totals['tool_failures']} failed)")
    table.add_row("duration", f"{summary['duration_s']}s")
    console.print(table)

    destination = out or Path(".forge") / "evals" / "report.json"
    write_report(report, destination)
    console.print(f"[dim]report written to {destination}[/dim]")


@app.command()
def sweep(
    tier: Optional[int] = typer.Option(None, help="Only run tier 1, 2 or 3."),
    model: Optional[str] = typer.Option(None, help="Model under test."),
    effort: str = typer.Option("smart", help="fast | smart | genius."),
    seeds: int = typer.Option(3, help="Repeats per task per configuration."),
    ablations: Optional[str] = typer.Option(
        None, help="Comma-separated ablation names (default: all)."
    ),
    out: Optional[Path] = typer.Option(None, help="Where to write the sweep JSON."),
) -> None:
    """Run the benchmark under every gate configuration — the ablation table."""
    import json as _json

    from forge.evals.runner import ABLATIONS, run_suite

    settings = _settings(model)
    settings.effort = effort
    names = (
        [a.strip() for a in ablations.split(",")] if ablations else list(ABLATIONS)
    )
    unknown = [a for a in names if a not in ABLATIONS]
    if unknown:
        console.print(f"[red]Unknown ablation(s):[/red] {', '.join(unknown)}")
        raise typer.Exit(2)

    console.print(
        Panel(
            f"model [bold]{settings.model}[/bold] · effort [bold]{effort}[/bold] · "
            f"seeds [bold]{seeds}[/bold] · configs [bold]{len(names)}[/bold]",
            title="SWE-micro ablation sweep",
        )
    )

    summaries: dict[str, dict] = {}
    for name in names:
        console.print(f"\n[bold]— {name} —[/bold]")
        report = run_suite(
            llm_factory=lambda: _client(settings),
            settings=settings,
            tier=tier,
            seeds=seeds,
            ablation=name,
            on_result=lambda r: console.print(
                f"  {'PASS' if r.solved else 'FAIL'} {r.task_id} "
                f"(T{r.tier}) {r.duration_s}s"
            ),
        )
        summaries[name] = report.summary()
        console.print(f"  [dim]TSR {report.task_success_rate() * 100:.1f}%[/dim]")

    # Persist BEFORE rendering: hours of runs must never be lost to a
    # console encoding error (a Windows cp1252 crash did exactly that once).
    destination = out or Path(".forge") / "evals" / "sweep.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    console.print(f"[dim]sweep written to {destination}[/dim]")

    table = Table("config", "TSR", "ADT", "FVR", "GER", "HIR", "WCR", "tools", "time")
    for name, summary in summaries.items():
        metrics = summary["metrics"]

        def show(field: str, m=metrics) -> str:
            value = m[field]["mean"]
            return "n/a" if value is None else f"{value * 100:.0f}%"

        table.add_row(
            name,
            f"{summary['task_success_rate'] * 100:.1f}%",
            show("act_dont_tell"),
            show("false_verification"),
            show("grounded_edit"),
            show("hallucinated_identifier"),
            show("wasted_cycle"),
            f"{metrics['totals']['tool_calls']}"
            f" ({metrics['totals']['tool_failures']} failed)",
            f"{summary['duration_s']:.0f}s",
        )
    console.print(table)


if __name__ == "__main__":
    app()
