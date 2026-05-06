"""
CLI runner for the 3-layer eval stack.

Usage:
    python scripts/run_evals.py run                    # Run all layers
    python scripts/run_evals.py run --layer 1          # Run Layer 1 only
    python scripts/run_evals.py run --category regulated
    python scripts/run_evals.py triage                 # Weekly failure triage
    python scripts/run_evals.py promote                # Promote queued failures

The three questions to answer before you ship another agent (from the article):
  1. Do you have a governed golden set owned by the business?
  2. Do you score with the right judgment pattern for the right risk?
  3. Does every production failure update your ground truth the same week?
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(help="3-Layer Eval Stack CLI")
console = Console()

# Add repo root to sys.path so imports work when run from scripts/
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


@app.command()
def run(
    layer: Optional[int] = typer.Option(None, help="Run only this layer (1, 2, or 3)"),
    category: Optional[str] = typer.Option(None, help="Filter golden set by category"),
    entry_id: Optional[str] = typer.Option(None, help="Run eval for a single golden entry ID"),
    judge: bool = typer.Option(False, help="Include LLM-as-judge (requires ANTHROPIC_API_KEY)"),
):
    """Run the eval stack against the golden set."""
    from evals.layer1_ground_truth.dataset import GoldenDataset
    from evals.layer1_ground_truth.metrics import run_layer1_metrics, layer1_passed

    console.print(Panel.fit(
        "[bold cyan]3-Layer Eval Stack[/bold cyan]\n"
        "Ground Truth → Judgment → Feedback",
        title="🔍 Eval Run",
    ))

    # Load golden set
    ds = GoldenDataset.load()
    if category:
        ds = ds.filter(category=category)
    if entry_id:
        ds = type(ds)([e for e in ds if e.id == entry_id])

    stats = ds.stats()
    console.print(f"\n[bold]Golden set:[/bold] {stats['total']} entries")
    for cat, count in stats["by_category"].items():
        console.print(f"  {cat}: {count}")

    if layer is None or layer == 1:
        _run_layer1(ds, run_layer1_metrics, layer1_passed)

    if (layer is None or layer == 2) and judge:
        _run_layer2_judge(ds)

    console.print("\n[bold green]✅ Eval run complete.[/bold green]")
    console.print(
        "\n[dim]Three questions before you ship another agent:[/dim]\n"
        "  1. Do you have a governed golden set owned by the business?\n"
        "  2. Do you score with the right judgment pattern for the right risk?\n"
        "  3. Does every production failure update your ground truth the same week?"
    )


def _run_layer1(ds, run_metrics, passed_fn):
    console.print("\n[bold yellow]─── Layer 1: Ground Truth ───[/bold yellow]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Entry ID", style="cyan", width=12)
    table.add_column("Category", width=20)
    table.add_column("must_contain", width=12)
    table.add_column("must_not_contain", width=16)
    table.add_column("token_f1", width=10)
    table.add_column("rouge_l", width=10)
    table.add_column("Overall", width=10)

    pass_count = 0
    for entry in ds:
        # For CLI demo, compare against the expected_answer itself (perfect score baseline)
        results = run_metrics(entry.expected_answer, entry)
        overall = passed_fn(results)
        if overall:
            pass_count += 1

        scores = {r.name: r for r in results}

        def fmt(name):
            r = scores.get(name)
            if r is None:
                return "[dim]n/a[/dim]"
            color = "green" if r.passed else "red"
            return f"[{color}]{r.score:.2f}[/{color}]"

        table.add_row(
            entry.id,
            entry.category,
            fmt("must_contain"),
            fmt("must_not_contain"),
            fmt("token_f1"),
            fmt("rouge_l"),
            "[green]PASS[/green]" if overall else "[red]FAIL[/red]",
        )

    console.print(table)
    total = len(ds)
    console.print(
        f"\nLayer 1: [bold]{pass_count}/{total}[/bold] entries passed "
        f"({'[green]100%[/green]' if pass_count == total else '[red]' + str(round(pass_count/total*100)) + '%[/red]'})"
    )


def _run_layer2_judge(ds):
    from evals.layer2_judgment.llm_judge.judge import ClaudeJudge
    console.print("\n[bold yellow]─── Layer 2: LLM-as-Judge ───[/bold yellow]")
    judge = ClaudeJudge()
    for entry in ds:
        if not entry.llm_judge_dimensions():
            continue
        console.print(f"  Evaluating [cyan]{entry.id}[/cyan]...")
        report = judge.evaluate(entry=entry, response=entry.expected_answer)
        status = "[green]PASS[/green]" if report.overall_passed else "[red]FAIL[/red]"
        console.print(f"    {status} weighted_avg={report.weighted_average:.3f}")


@app.command()
def triage(n: int = typer.Option(50, help="Number of traces to sample")):
    """Weekly failure triage: sample production traces and cluster failures."""
    from evals.layer3_feedback.collector import ProductionFeedbackCollector
    from evals.layer3_feedback.analyzer import FailurePatternAnalyzer

    console.print(Panel.fit("[bold]Weekly Failure Triage[/bold]", title="🔎 Layer 3"))

    collector = ProductionFeedbackCollector()
    analyzer = FailurePatternAnalyzer()

    stats = collector.stats()
    console.print(f"\n[bold]Feedback log:[/bold] {stats['total_traces']} total traces")
    console.print(f"Failure candidates: {stats['failure_candidates']}")
    console.print(f"Already promoted: {stats['promoted_to_golden']}")

    candidates = collector.failure_candidates()
    if not candidates:
        console.print("\n[green]No failure candidates in the log. All good![/green]")
        return

    summaries = analyzer.cluster_batch(candidates)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Cluster", style="cyan", width=28)
    table.add_column("Count", width=8)
    table.add_column("Owner", width=18)
    table.add_column("Example", width=50)

    for s in summaries:
        example = s.example_questions[0] if s.example_questions else ""
        table.add_row(s.cluster.value, str(s.count), s.routing_owner, example[:48])

    console.print(table)
    console.print(
        f"\n[yellow]Next step:[/yellow] For each cluster, promote confirmed failures to the golden set.\n"
        f"Run [bold]python scripts/run_evals.py promote[/bold] after human review."
    )


@app.command()
def status():
    """Print a status report answering the article's three production-readiness questions."""
    from evals.layer1_ground_truth.dataset import GoldenDataset
    from evals.layer2_judgment.human_review.review_queue import HumanReviewQueue
    from evals.layer3_feedback.collector import ProductionFeedbackCollector

    ds = GoldenDataset.load()
    queue = HumanReviewQueue()
    collector = ProductionFeedbackCollector()

    ds_stats = ds.stats()
    queue_stats = queue.stats()
    collector_stats = collector.stats()

    console.print(Panel.fit("[bold]Production Readiness Report[/bold]", title="📊 Eval Stack Status"))
    console.print()

    # Q1: Governed golden set?
    has_all_categories = all(
        cat in ds_stats["by_category"]
        for cat in ["regulated", "historical-failures", "adversarial"]
    )
    q1_ok = ds_stats["total"] > 0 and has_all_categories
    console.print(
        f"{'[green]✅[/green]' if q1_ok else '[red]❌[/red]'} "
        f"[bold]Q1:[/bold] Governed golden set? "
        f"{ds_stats['total']} entries | "
        + ", ".join(f"{k}: {v}" for k, v in ds_stats["by_category"].items())
    )

    # Q2: Right judgment patterns?
    require_human = ds_stats.get("require_human_review", 0)
    console.print(
        f"[green]✅[/green] [bold]Q2:[/bold] Judgment patterns active | "
        f"{require_human} entries require human review"
    )

    # Q3: Feedback loop active?
    q3_ok = collector_stats["total_traces"] > 0
    console.print(
        f"{'[green]✅[/green]' if q3_ok else '[yellow]⚠️[/yellow]'} "
        f"[bold]Q3:[/bold] Feedback loop | "
        f"{collector_stats['total_traces']} traces logged | "
        f"{collector_stats['promoted_to_golden']} promoted | "
        f"{queue_stats['total']} in review queue"
    )


if __name__ == "__main__":
    app()
