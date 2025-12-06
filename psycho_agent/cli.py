"""Typer-based CLI to run the Psycho-World agent."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from .workflow import PsychoWorldGraph

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def chat(user_input: str, user_id: str = "demo") -> None:
    """Invoke the multi-agent workflow once."""
    agent = PsychoWorldGraph()
    result = agent.invoke(user_input=user_input, user_id=user_id)
    console.print(result["final_response"])
    console.print("\n[bold blue]Diagnostics[/bold blue]")
    console.print_json(data=result.get("diagnostics", {}))


if __name__ == "__main__":
    app()
