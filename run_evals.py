#!/usr/bin/env python3
"""
run_evals.py – Config-driven CLI for the 3-layer LLM evaluation stack.

Commands:
  run     [--id ID] [--category CAT] [--report FORMAT] [--report-dir DIR]
  status                               show feedback log stats
  triage                               cluster recent failure candidates
  promote --id TRACE_ID --answer TEXT [--write]  promote a failure to the golden set

Examples:
  python run_evals.py run
  python run_evals.py run --id faq-reg-001
  python run_evals.py run --category regulated --report json html --report-dir reports
  python run_evals.py status
  python run_evals.py triage
  python run_evals.py promote --id trace-faq-reg-001 --answer "30 days policy." --write
"""
from __future__ import annotations

import argparse
import importlib
import io
import sys
import warnings
from pathlib import Path

# UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evals.config import load_eval_config, EVAL_CONFIG, REPORTS_DIR
from evals.layer1_ground_truth.dataset import GoldenDataset, GoldenEntry
from evals.reporter import generate_report
from evals.runner import EvalRunner
from evals.token_tracker import render_token_table


# ── Built-in placeholder agent ─────────────────────────────────────────────────

def _builtin_faq_agent(question: str, context: str) -> str:
    """
    Canned answer for the demo golden entry.
    Swap this out via eval_config.yaml agent.module / agent.function.
    """
    return (
        "You can return electronics within 30 days of purchase. "
        "Please bring your original receipt and ensure the item is in its "
        "original packaging. Your refund will be processed within 5-7 business "
        "days back to your original payment method."
    )


def load_agent_fn(cfg: dict):
    """
    Resolve the agent callable from eval_config.yaml or fall back to the
    built-in placeholder.

    Config keys (under 'agent:'):
        module:   dotted importable Python path, e.g. "my_package.handler"
        function: callable name inside that module (default: "generate_answer")

    Example eval_config.yaml snippet:
        agent:
          module: my_agent.handler
          function: generate_answer
    """
    agent_cfg   = cfg.get("agent", {})
    module_path = agent_cfg.get("module")
    fn_name     = agent_cfg.get("function", "generate_answer")

    if not module_path:
        return _builtin_faq_agent

    try:
        mod = importlib.import_module(module_path)
        fn  = getattr(mod, fn_name)
        print(f"  [agent] loaded {module_path}.{fn_name}")
        return fn
    except (ImportError, AttributeError) as exc:
        print(f"  [agent] WARNING: could not import {module_path}.{fn_name}: {exc}")
        print("  [agent] falling back to built-in placeholder.")
        return _builtin_faq_agent


# ── Display helpers ────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    w = 68
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def _print_result(result, verbose: bool = True) -> None:
    icon = "[OK]" if result.all_passed else "[!!]"
    print(f"\n  {icon} {result.entry_id}  -  {result.question[:60]}")

    if not verbose:
        print(f"       {result.summary()}")
        return

    # Layer 1
    print(f"\n  -- Layer 1  Ground Truth --")
    for m in result.metrics:
        ok  = "[OK]" if m.passed else "[!!]"
        bar = "#" * int(m.score * 10) + "." * (10 - int(m.score * 10))
        print(f"  {ok} {m.name:<22} [{bar}] {m.score:.3f}  {m.details}")
    print(f"  => {'PASS' if result.l1_passed else 'FAIL'}")

    # Layer 2a
    print(f"\n  -- Layer 2a  Code Evaluators --")
    for r in result.code_results:
        ok = "[OK]" if r.passed else "[!!]"
        print(f"  {ok} {r.evaluator:<42}  {r.reason}")

    # Layer 2b
    print(f"\n  -- Layer 2b  LLM Judge --")
    print(result.judge_report.summary())

    # Layer 2 verdict
    print(f"\n  Layer 2 => {'PASS' if result.l2_passed else 'FAIL'}")

    # Layer 3
    print(f"\n  -- Layer 3  Feedback --")
    print(f"  Trace logged  signal={'POSITIVE' if result.l2_passed else 'LLM_JUDGE_LOW'}"
          f"  judge={result.judge_report.weighted_average:.3f}")
    if result.failure_cluster:
        print(f"  Clustered as: {result.failure_cluster}")

    # Tokens
    print()
    print(render_token_table(result.token_layers))


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    cfg      = load_eval_config(Path(args.config))
    runner   = EvalRunner(config_path=args.config)
    agent_fn = load_agent_fn(cfg)

    dataset = GoldenDataset(
        golden_set_path=Path(cfg.get("golden_set", {}).get(
            "path", "evals/layer1_ground_truth/golden_set"
        ))
    )
    try:
        dataset.load()
        entries = dataset.entries
    except Exception:
        entries = []

    if not entries:
        entries = [_demo_entry(cfg)]

    if args.id:
        entries = [e for e in entries if e.id == args.id]
        if not entries:
            print(f"  No golden entry with id '{args.id}' found.")
            return 1
    elif args.category:
        entries = [e for e in entries if e.category == args.category]
        if not entries:
            print(f"  No entries in category '{args.category}'.")
            return 1

    _section(f"run  ({len(entries)} entr{'y' if len(entries)==1 else 'ies'})")

    passed = failed = 0
    all_results = []
    all_tokens  = []
    for entry in entries:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = runner.run(entry, agent_fn=agent_fn)
        _print_result(result, verbose=not args.summary)
        all_results.append(result)
        all_tokens.extend(result.token_layers)
        if result.all_passed:
            passed += 1
        else:
            failed += 1

    _section("Summary")
    print(f"\n  Passed : {passed}   Failed : {failed}   Total : {passed + failed}")
    if len(entries) > 1:
        print()
        print(render_token_table(all_tokens))

    # Report generation
    if args.report:
        out_dir = Path(args.report_dir)
        paths   = generate_report(all_results, out_dir=out_dir, formats=args.report)
        print(f"\n  Reports written:")
        for fmt, p in paths.items():
            print(f"    [{fmt.upper()}] {p}")

    return 0 if failed == 0 else 1


def cmd_status(args: argparse.Namespace) -> int:
    _section("status  -  feedback log")
    runner = EvalRunner(config_path=args.config)
    stats  = runner.collector.stats(from_disk=True)
    print(f"\n  Total traces      : {stats['total_traces']}")
    print(f"  Failure candidates: {stats['failure_candidates']}")
    print(f"  By signal:")
    for sig, count in stats["by_signal"].items():
        print(f"    {sig:<20} {count}")
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    _section("triage  -  failure clusters")
    runner   = EvalRunner(config_path=args.config)
    clusters = runner.triage(from_disk=True)
    if not clusters:
        print("\n  No failure candidates found.")
        return 0
    for cluster, info in clusters.items():
        print(f"\n  [{cluster}]  count={info['count']}  team={info['team']}")
        print(f"    sample Q: \"{info['sample_question']}\"")
        print(f"    traces  : {info['trace_ids']}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    _section("promote  -  failure -> golden set")
    runner   = EvalRunner(config_path=args.config)
    promoted = runner.promote_failures({args.id: args.answer}, write=args.write)
    if not promoted:
        print(f"\n  No failure candidate with trace_id '{args.id}' found.")
        return 1
    from evals.layer3_feedback.updater import GoldenSetUpdater
    for entry in promoted:
        status = "written to disk" if args.write else "preview only (use --write to persist)"
        print(f"\n  Promoted: {entry['id']}  ({entry['category']})  [{status}]")
        print()
        for line in GoldenSetUpdater.to_yaml_str(entry).split("\n"):
            print(f"    {line}")
    return 0


def _demo_entry(cfg: dict) -> GoldenEntry:
    """Inline golden entry used when no YAML files are present."""
    return GoldenEntry(
        id="faq-reg-001",
        category="regulated",
        question="What is your return policy for electronics?",
        context="",
        expected_answer=(
            "Electronics can be returned within 30 days with original receipt "
            "and packaging. Refunds take 5-7 business days."
        ),
        must_contain=["30 days", "original receipt", "5-7 business days"],
        must_not_contain=["lifetime", "90 days", "immediate refund"],
        judgment_patterns=["code_evaluator", "llm_judge"],
    )


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_evals",
        description="3-Layer LLM Evaluation Stack - CLI runner",
    )
    p.add_argument("--config", default=str(EVAL_CONFIG),
                   help="Path to eval_config.yaml (default: %(default)s)")

    sub = p.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Evaluate golden entries")
    run_p.add_argument("--id",       help="Run a single entry by ID")
    run_p.add_argument("--category", help="Filter by category (regulated|historical-failures|adversarial)")
    run_p.add_argument("--summary",  action="store_true",
                       help="One-line output per entry (no per-layer detail)")
    run_p.add_argument("--report",   nargs="+", choices=["json", "csv", "html"],
                       metavar="FORMAT",
                       help="Write a structured report: json, csv, and/or html")
    run_p.add_argument("--report-dir", default=str(REPORTS_DIR),
                       help="Directory to write reports into (default: %(default)s)")

    # status
    sub.add_parser("status", help="Show feedback log statistics")

    # triage
    sub.add_parser("triage", help="Cluster recent failure candidates")

    # promote
    prom_p = sub.add_parser("promote", help="Promote a failure to the golden set")
    prom_p.add_argument("--id",     required=True, help="trace_id to promote")
    prom_p.add_argument("--answer", required=True, help="Validated correct answer")
    prom_p.add_argument("--write",  action="store_true",
                        help="Persist the promoted entry as a YAML file in the golden set")

    return p


def main() -> None:
    parser   = build_parser()
    args     = parser.parse_args()
    dispatch = {
        "run":     cmd_run,
        "status":  cmd_status,
        "triage":  cmd_triage,
        "promote": cmd_promote,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
