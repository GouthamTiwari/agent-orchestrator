"""Command line entry point (`orchestrator`). Commands are added task by task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agent_orchestrator import __version__

app = typer.Typer(help="GST ITC reconciliation agents.", no_args_is_help=True)
dataset_app = typer.Typer(help="Generate and review the evaluation dataset.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


@app.callback()
def main() -> None:
    """GST ITC reconciliation agents."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@dataset_app.command("generate")
def dataset_generate(
    out: Annotated[
        Path | None, typer.Option(help="Output folder (default data/generated/).")
    ] = None,
    render: Annotated[bool, typer.Option(help="Render invoice PDFs and noisy scans.")] = True,
) -> None:
    """Write the 10-business x 3-month dataset and refresh the committed labels."""
    from agent_orchestrator.eval import generate as gen

    summary = gen.generate(out_dir=out or gen.DEFAULT_OUT, render=render)
    typer.echo(
        f"{summary['business_months']} business-months, {summary['labels']} labelled cases, "
        f"{summary['invoices']} invoices ({summary['noisy']} noisy scans), "
        f"{summary['injections']} injection invoices -> {out or gen.DEFAULT_OUT}"
    )


@dataset_app.command("split")
def dataset_split() -> None:
    """Check data/splits.json: every business in exactly one split, no supplier in both."""
    from agent_orchestrator.eval import generate as gen

    splits = json.loads((gen.ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
    report = check_splits(splits, gen.build_businesses(gen.load_config()))
    for name in ("dev", "test"):
        typer.echo(f"{name}: {', '.join(splits[name])} ({report[name]} suppliers)")
    typer.echo("OK: no business or supplier GSTIN appears in both splits")


def check_splits(splits: dict, businesses: list) -> dict[str, int]:
    """Validate the split; raise typer.BadParameter on overlap. Returns suppliers per split."""
    dev, test = set(splits["dev"]), set(splits["test"])
    ids = {b.business_id for b in businesses}
    if dev & test or (dev | test) != ids:
        raise typer.BadParameter("each business must be in exactly one of dev and test")
    gstins = {
        name: {s.gstin for b in businesses if b.business_id in members for s in b.suppliers}
        | {b.gstin for b in businesses if b.business_id in members}
        for name, members in (("dev", dev), ("test", test))
    }
    if gstins["dev"] & gstins["test"]:
        raise typer.BadParameter("a GSTIN appears in both splits")
    return {name: len(g) - len(splits[name]) for name, g in gstins.items()}


@dataset_app.command("review-sample")
def dataset_review_sample(
    out: Annotated[Path | None, typer.Option(help="CSV to write (default data/review/).")] = None,
) -> None:
    """Write a 10% sample of labels, stratified by case type, for a human spot-check."""
    from agent_orchestrator.eval import generate as gen
    from agent_orchestrator.eval import review

    target = out or review.DEFAULT_SAMPLE
    n = review.write_sample(gen.load_labels(), target)
    typer.echo(
        f"{n} labels written to {target}. Fill 'verdict' with OK or FIX, then run "
        "'orchestrator dataset apply-review FILE'."
    )


@dataset_app.command("apply-review")
def dataset_apply_review(filled: Path) -> None:
    """Apply a filled review CSV: record overrides and log the changes."""
    from agent_orchestrator.eval import generate as gen
    from agent_orchestrator.eval import review

    try:
        result = review.apply_review(filled, gen.load_labels())
    except review.ReviewError as exc:
        typer.echo(f"Nothing applied. Problems:\n{exc}", err=True)
        raise typer.Exit(1) from exc
    gen.generate(render=False)  # rewrite labels with the overrides applied
    typer.echo(
        f"{result['reviewed']} reviewed, {result['fixed']} fixed. "
        "Labels regenerated; see docs/DATASET_CHANGELOG.md."
    )


if __name__ == "__main__":
    app()
