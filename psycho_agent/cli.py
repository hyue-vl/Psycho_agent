"""Typer-based CLI to run the Psycho-World agent."""

from __future__ import annotations

import typer
from rich.console import Console

from .workflow import PsychoWorldGraph

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_EXIT_TOKENS = {"exit", "quit", "q"}
WORKFLOW_KEYS = [
    ("perception", "Perception"),
    ("planning_prompt", "Planning Prompt"),
    ("planning_tree", "Planning Tree"),
    ("simulation_outputs", "Simulation Scorecards"),
    ("simulation_rollouts", "Simulation Rollouts"),
    ("action_prompt", "Action Prompt"),
]


def _render_response(turn: int, payload: dict, show_workflow: bool, show_diagnostics: bool) -> None:
    response = (payload.get("final_response") or "").strip()
    if not response:
        response = "[no response generated]"
    console.print(f"\n[bold cyan]Agent[{turn}][/bold cyan] {response}")
    diagnostics = payload.get("diagnostics", {}) or {}
    if show_workflow:
        _render_workflow_trace(diagnostics)
    if show_diagnostics:
        console.print("\n[bold blue]Diagnostics[/bold blue]")
        console.print_json(data=diagnostics)


def _render_workflow_trace(diagnostics: dict) -> None:
    console.print("\n[bold magenta]Workflow Trace[/bold magenta]")
    if not diagnostics:
        console.print("[dim]No diagnostics emitted for this turn.[/dim]")
        return
    seen = set()
    for key, label in WORKFLOW_KEYS:
        if key not in diagnostics:
            continue
        _pretty_print_section(label, diagnostics[key])
        seen.add(key)
    remaining = {k: v for k, v in diagnostics.items() if k not in seen}
    if remaining:
        _pretty_print_section("Additional Diagnostics", remaining)


def _pretty_print_section(title: str, payload: object) -> None:
    console.print(f"[bold]{title}[/bold]")
    if isinstance(payload, str):
        console.print(payload)
    else:
        try:
            console.print_json(data=payload)
        except Exception:
            console.print(repr(payload))


def _exit_tokens(custom: str | None) -> set[str]:
    tokens = {token.lower() for token in DEFAULT_EXIT_TOKENS}
    if custom:
        tokens.add(custom.lower())
    return tokens


def _should_exit(user_text: str, tokens: set[str]) -> bool:
    return user_text.strip().lower() in tokens


def _process_turn(
    agent: PsychoWorldGraph,
    turn_index: int,
    user_input: str,
    user_id: str,
    show_workflow: bool,
    show_diagnostics: bool,
) -> None:
    result = agent.invoke(user_input=user_input, user_id=user_id)
    _render_response(turn_index, result, show_workflow, show_diagnostics)


@app.command()
def chat(
    opening: str | None = typer.Argument(
        None,
        help="Optional first user utterance; conversation continues interactively.",
    ),
    user_id: str = typer.Option("demo", "--user-id", help="Unique identifier for memory scoping."),
    max_turns: int = typer.Option(
        10,
        "--max-turns",
        min=0,
        help="Maximum user turns before auto-exit (0 means unlimited).",
    ),
    workflow: bool = typer.Option(
        True,
        "--workflow/--no-workflow",
        help="Print a structured workflow trace for each turn.",
    ),
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics/--no-diagnostics",
        help="Print diagnostics JSON after each turn.",
    ),
    exit_phrase: str = typer.Option(
        "exit",
        "--exit-phrase",
        help="Custom phrase to terminate the session.",
    ),
) -> None:
    """Run the Psycho-World agent in a multi-turn conversational loop."""
    agent = PsychoWorldGraph()
    console.print("[bold green]Psycho-World Conversational Agent[/bold green]")
    console.print(f"[dim]Type your thoughts. Enter '{exit_phrase}' or press Ctrl-D to stop.[/dim]")
    tokens = _exit_tokens(exit_phrase)
    limit = max_turns if max_turns > 0 else None
    turn = 0

    def _respect_limit(current_turn: int) -> bool:
        return limit is not None and current_turn >= limit

    if opening:
        turn += 1
        _process_turn(agent, turn, opening, user_id, workflow, diagnostics)
        if _respect_limit(turn):
            console.print("[yellow]Max turn limit reached.[/yellow]")
            return

    while True:
        if _respect_limit(turn):
            console.print("[yellow]Max turn limit reached.[/yellow]")
            break
        try:
            user_input = typer.prompt(f"User[{turn + 1}]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Session interrupted by user.[/red]")
            break
        if not user_input:
            continue
        if _should_exit(user_input, tokens):
            console.print("[dim]Session ended by user.[/dim]")
            break
        turn += 1
        _process_turn(agent, turn, user_input, user_id, workflow, diagnostics)


if __name__ == "__main__":
    app()
