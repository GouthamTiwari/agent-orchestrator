"""Command line entry point (`orchestrator`). Commands are added task by task."""

import typer

from agent_orchestrator import __version__

app = typer.Typer(help="GST ITC reconciliation agents.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """GST ITC reconciliation agents."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
