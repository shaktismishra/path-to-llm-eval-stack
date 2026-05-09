"""
Structured report generation for eval runs.

Supports JSON, CSV, and HTML output formats.
Reports are written to reports/ by default (configurable via --report-dir).

Usage:
    from evals.reporter import generate_report
    paths = generate_report(results, out_dir=Path("reports"), formats=["json", "html"])
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ReportRow:
    entry_id:        str
    question:        str
    verdict:         str   # "PASS" | "FAIL"
    l1_passed:       bool
    l2_passed:       bool
    judge_score:     float
    failure_cluster: str | None
    total_tokens:    int


def _build_rows(results: list) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for r in results:
        rows.append(ReportRow(
            entry_id        = r.entry_id,
            question        = r.question[:120],
            verdict         = "PASS" if r.all_passed else "FAIL",
            l1_passed       = r.l1_passed,
            l2_passed       = r.l2_passed,
            judge_score     = round(r.judge_report.weighted_average, 4),
            failure_cluster = r.failure_cluster,
            total_tokens    = sum(tl.total for tl in r.token_layers),
        ))
    return rows


def _summary(rows: list[ReportRow]) -> dict:
    passed = sum(1 for r in rows if r.verdict == "PASS")
    return {
        "total":  len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "avg_judge_score": round(
            sum(r.judge_score for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "total_tokens": sum(r.total_tokens for r in rows),
    }


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ── Format writers ─────────────────────────────────────────────────────────────

def write_json(rows: list[ReportRow], out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_report_{run_id}.json"
    payload = {
        "run_id":       run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary":      _summary(rows),
        "entries":      [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_csv(rows: list[ReportRow], out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_report_{run_id}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(asdict(rows[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return path


def write_html(rows: list[ReportRow], out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_report_{run_id}.html"
    s = _summary(rows)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    table_rows = ""
    for r in rows:
        bg      = "#d4edda" if r.verdict == "PASS" else "#f8d7da"
        cluster = r.failure_cluster or "-"
        q       = r.question[:70] + ("..." if len(r.question) > 70 else "")
        table_rows += (
            f'<tr style="background:{bg}">'
            f"<td>{r.entry_id}</td>"
            f"<td>{q}</td>"
            f"<td><b>{r.verdict}</b></td>"
            f"<td>{'Yes' if r.l1_passed else 'No'}</td>"
            f"<td>{'Yes' if r.l2_passed else 'No'}</td>"
            f"<td>{r.judge_score:.3f}</td>"
            f"<td>{cluster}</td>"
            f"<td>{r.total_tokens:,}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Eval Report {run_id}</title>
  <style>
    body  {{ font-family: sans-serif; margin: 2em; color: #333 }}
    h1    {{ color: #343a40 }}
    .cards {{ display: flex; gap: 1.5em; margin: 1em 0 1.5em }}
    .card {{ padding: .8em 1.6em; border-radius: 6px; color: #fff; font-size: 1.1em }}
    .total {{ background: #6c757d }} .pass {{ background: #28a745 }}
    .fail  {{ background: #dc3545 }} .rate {{ background: #17a2b8 }}
    table  {{ border-collapse: collapse; width: 100%; font-size: .9em }}
    th,td  {{ border: 1px solid #dee2e6; padding: 6px 10px; text-align: left }}
    th     {{ background: #343a40; color: #fff }}
  </style>
</head>
<body>
<h1>Eval Report <code>{run_id}</code></h1>
<p>Generated {ts} &nbsp;|&nbsp; Avg judge score: <b>{s['avg_judge_score']:.3f}</b>
   &nbsp;|&nbsp; Total tokens: <b>{s['total_tokens']:,}</b></p>
<div class="cards">
  <div class="card total">Total: {s['total']}</div>
  <div class="card pass">Passed: {s['passed']}</div>
  <div class="card fail">Failed: {s['failed']}</div>
  <div class="card rate">Pass rate: {s['pass_rate']*100:.0f}%</div>
</div>
<table>
  <tr>
    <th>ID</th><th>Question</th><th>Verdict</th>
    <th>L1</th><th>L2</th><th>Judge</th><th>Cluster</th><th>Tokens</th>
  </tr>
  {table_rows}
</table>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_report(
    results: list,
    out_dir: Path,
    formats: list[str],
) -> dict[str, Path]:
    """
    Write reports in the requested formats.

    Args:
        results:  list of EvalResult objects from EvalRunner.run()
        out_dir:  directory to write reports into
        formats:  subset of ["json", "csv", "html"]

    Returns:
        dict mapping format name to the written file Path.
    """
    rows   = _build_rows(results)
    rid    = _run_id()
    output: dict[str, Path] = {}
    for fmt in formats:
        if fmt == "json":
            output["json"] = write_json(rows, out_dir, rid)
        elif fmt == "csv":
            output["csv"]  = write_csv(rows, out_dir, rid)
        elif fmt == "html":
            output["html"] = write_html(rows, out_dir, rid)
    return output
