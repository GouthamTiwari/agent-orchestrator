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


def _print_result(result) -> None:
    typer.echo(f"run {result.run_id}: {result.status}")
    if result.pending:
        typer.echo(
            f"{len(result.pending)} approvals waiting. Decide them, then run "
            f"'orchestrator resume {result.run_id}'."
        )
    typer.echo(f"report: {result.report}")


@app.command()
def run(
    demo: Annotated[
        bool, typer.Option(help="The seed month with the fake model (no AWS).")
    ] = False,
    data: Annotated[
        Path | None,
        typer.Option(help="Month folder: purchase_register.csv, gstr2b_like.json, invoices/."),
    ] = None,
    period: Annotated[str | None, typer.Option(help="YYYY-MM.")] = None,
    business: Annotated[str, typer.Option(help="Business id.")] = "B",
    live: Annotated[bool, typer.Option(help="Call the real model (costs money).")] = False,
    provider: Annotated[str, typer.Option(help="bedrock or anthropic (with --live).")] = "bedrock",
    yes: Annotated[bool, typer.Option(help="Do not ask before a live run.")] = False,
    rag: Annotated[bool, typer.Option(help="Use Regulatory RAG if it is available.")] = False,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: generated).")] = None,
    db: Annotated[Path | None, typer.Option(help="SQLite file.")] = None,
    runs_dir: Annotated[Path | None, typer.Option(help="Where reports go.")] = None,
) -> None:
    """Run one business-month until the approval gate (or the end)."""
    from agent_orchestrator.orchestrator import runner

    if demo:
        if live:
            raise typer.BadParameter("--demo always uses the fake model; drop --live")
        cfg = runner.demo_config(run_id, use_rag=rag)
    else:
        if data is None or period is None:
            raise typer.BadParameter("give --demo, or --data and --period")
        cfg = runner.RunConfig(
            run_id=run_id or runner.new_run_id(business, period),
            business_id=business, period=period,
            books_path=str(data / "purchase_register.csv"),
            portal_path=str(data / "gstr2b_like.json"),
            invoices_dir=str(data / "invoices") if (data / "invoices").is_dir() else None,
            mode="live" if live else "replay", provider=provider, use_rag=rag,
        )  # fmt: skip
        if live and not yes:
            typer.echo(
                f"LIVE run with {provider}: real model calls, capped by the run budget "
                f"(USD {cfg.budget_usd:.2f}) and the project cap."
            )
            if not typer.confirm("Continue?", default=False):
                raise typer.Exit(1)
    result = runner.start_run(
        cfg, db_path=db or runner.DEFAULT_DB, runs_dir=runs_dir or runner.DEFAULT_RUNS
    )
    _print_result(result)


@app.command()
def resume(
    run_id: str,
    db: Annotated[Path | None, typer.Option(help="SQLite file.")] = None,
    runs_dir: Annotated[Path | None, typer.Option(help="Where reports go.")] = None,
) -> None:
    """Continue a paused or interrupted run from its last checkpoint."""
    from agent_orchestrator.orchestrator import runner

    try:
        result = runner.resume_run(
            run_id, db_path=db or runner.DEFAULT_DB, runs_dir=runs_dir or runner.DEFAULT_RUNS
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _print_result(result)


approvals_app = typer.Typer(help="Decide what the policy sent to a person.", no_args_is_help=True)
app.add_typer(approvals_app, name="approvals")
DbOption = Annotated[Path | None, typer.Option("--db", help="SQLite file.")]
ByOption = Annotated[
    str | None, typer.Option("--by", help="Your name (default APPROVER_NAME, else OS user).")
]


def _open_repo(db: Path | None):
    from agent_orchestrator.orchestrator import runner
    from agent_orchestrator.store.repo import Repo

    path = db or runner.DEFAULT_DB
    if not Path(path).exists():
        raise typer.BadParameter(f"no database at {path}; run 'orchestrator run' first")
    return Repo.open(path)


def _human(fn, *args, **kwargs) -> None:
    from agent_orchestrator.tools import toolfns

    try:
        done = fn(*args, **kwargs)
    except toolfns.HumanActionError as exc:
        typer.echo(f"Not done: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"{done.approval_id}: {done.status} -> {done.final_action} by {done.decided_by}. "
        f"When nothing is pending, run 'orchestrator resume {done.run_id}'."
    )


@approvals_app.command("list")
def approvals_list(
    run_id: Annotated[str | None, typer.Option("--run", help="Only this run.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Include decided ones.")] = False,
    db: DbOption = None,
) -> None:
    """Show approvals waiting for a person (oldest first)."""
    repo = _open_repo(db)
    try:
        rows = repo.list_approvals(None if all_ else "pending", run_id)
        for a in rows:
            case = repo.conn.execute(
                "SELECT mismatch_type FROM cases WHERE case_key = ?", (a.case_key,)
            ).fetchone()
            kind = case[0] if case else "?"
            done = (
                f" -> {a.final_action} ({a.status}, {a.decided_by})"
                if a.status != "pending"
                else ""
            )
            typer.echo(
                f"{a.approval_id}  run {a.run_id}  {kind}  proposed {a.proposed}{done}  "
                f"options: {', '.join(a.allowed_options)}"
            )
        typer.echo(f"{len(rows)} {'approvals' if all_ else 'pending'}")
    finally:
        repo.close()


@approvals_app.command("approve")
def approvals_approve(approval_id: str, by: ByOption = None, db: DbOption = None) -> None:
    """Accept the proposed action."""
    from agent_orchestrator.tools import toolfns

    repo = _open_repo(db)
    try:
        _human(toolfns.approve, repo, approval_id, by=by)
    finally:
        repo.close()


@approvals_app.command("edit")
def approvals_edit(
    approval_id: str,
    action: Annotated[str, typer.Argument(help="One of the allowed options.")],
    reason: Annotated[str, typer.Option(help="Why (required).")] = "",
    by: ByOption = None,
    db: DbOption = None,
) -> None:
    """Choose a different allowed action."""
    from agent_orchestrator.tools import toolfns

    repo = _open_repo(db)
    try:
        _human(toolfns.edit, repo, approval_id, action, reason=reason, by=by)
    finally:
        repo.close()


@approvals_app.command("reject")
def approvals_reject(
    approval_id: str,
    reason: Annotated[str, typer.Option(help="Why (required).")] = "",
    by: ByOption = None,
    db: DbOption = None,
) -> None:
    """Say no: the case stays with a person (escalate_human)."""
    from agent_orchestrator.tools import toolfns

    repo = _open_repo(db)
    try:
        _human(toolfns.reject, repo, approval_id, reason=reason, by=by)
    finally:
        repo.close()


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
