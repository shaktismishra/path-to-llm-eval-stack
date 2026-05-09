#!/usr/bin/env python3
"""
demo_eval_agent.py – Quick-start demo using EvalRunner.

For the full config-driven CLI use:  python run_evals.py run

Run:  python -X utf8 demo_eval_agent.py
"""
from __future__ import annotations
import sys, io, warnings

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evals.runner import EvalRunner
from evals.layer1_ground_truth.dataset import GoldenEntry
from evals.layer3_feedback.collector import ProductionTrace, FeedbackSignal
from evals.layer3_feedback.updater import GoldenSetUpdater
from evals.token_tracker import render_token_table
from evals.config import JUDGE_THRESHOLD


# ── Agents ─────────────────────────────────────────────────────────────────────

def faq_agent(question: str, context: str) -> str:
    """Good agent response – PASSES all layers."""
    return (
        "You can return electronics within 30 days of purchase. "
        "Please bring your original receipt and ensure the item is in its "
        "original packaging. Your refund will be processed within 5-7 business "
        "days back to your original payment method."
    )


def faq_agent_bad(question: str, context: str) -> str:
    """Broken agent response – feeds Layer 3 feedback demo."""
    return (
        "We offer next-day delivery on all electronics! "
        "You can return items anytime within 90 days -- no receipt needed. "
        "Refunds are instant."
    )


# ── Display ─────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _verdict(label: str, passed: bool, note: str = "") -> None:
    v = "PASS" if passed else "FAIL"
    print(f"\n  +-- {label} : {v} " + "-" * (50 - len(label)) + "+")
    if note:
        print(f"  |  {note}")
    print(f"  +" + "-" * 65 + "+")


# ── Demo ────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    print()
    print("=" * 70)
    print("  3-Layer LLM Evaluation Stack  --  Demo")
    print("  Domain : FAQ Knowledge Base Bot")
    print("  Runner : EvalRunner (eval_config.yaml)")
    print("=" * 70)

    golden = GoldenEntry(
        id="faq-reg-001",
        category="regulated",
        question="What is your return policy for electronics?",
        context="",   # EvalRunner retrieves this from docs/faq.md
        expected_answer=(
            "Electronics can be returned within 30 days with original receipt "
            "and packaging. Refunds take 5-7 business days."
        ),
        must_contain=["30 days", "original receipt", "5-7 business days"],
        must_not_contain=["lifetime", "90 days", "immediate refund"],
        judgment_patterns=["code_evaluator", "llm_judge"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = EvalRunner(persist=False)   # no disk writes in demo mode
        result = runner.run(golden, agent_fn=faq_agent)

    print(f"\n  Golden entry : {result.entry_id}")
    print(f"  Question     : {result.question}")
    print(f"  Retrieved    : {len(result.context)} chars from docs/faq.md")
    print(f"\n  Agent response:")
    print(f"  \"{result.response}\"")

    # ── Layer 1 ────────────────────────────────────────────────────────────────
    _section("LAYER 1 -- Ground Truth  (Deterministic Metrics)")
    print("  evals/layer1_ground_truth/metrics.py\n")
    for m in result.metrics:
        bar = "#" * int(m.score * 10) + "." * (10 - int(m.score * 10))
        ok  = "[OK]" if m.passed else "[!!]"
        print(f"  {ok} {m.name:<22} [{bar}] {m.score:.3f} >= {m.threshold}   {m.details}")
    _verdict("Layer 1 verdict", result.l1_passed,
             "All constraint gates passed + at least one similarity metric >= threshold")
    assert result.l1_passed

    # ── Layer 2 ────────────────────────────────────────────────────────────────
    _section("LAYER 2 -- Judgment Patterns")

    print("\n  -- 2a  Code Evaluators  (evals/layer2_judgment/code_evaluators/evaluators.py) --")
    for r in result.code_results:
        ok = "[OK]" if r.passed else "[!!]"
        print(f"  {ok} {r.evaluator:<42}  {r.reason}")
    l2a = all(r.passed for r in result.code_results)
    print(f"\n  Code evaluators : {'all PASS' if l2a else 'FAIL'}")

    print("\n  -- 2b  LLM-as-Judge  (evals/layer2_judgment/llm_judge/judge.py) --")
    print("  [production: ClaudeJudge  |  demo: MockClaudeJudge via get_judge()]\n")
    print(result.judge_report.summary())

    _verdict("Layer 2 verdict", result.l2_passed,
             f"Code evaluators clean + judge score "
             f"{result.judge_report.weighted_average:.3f} >= {JUDGE_THRESHOLD}")
    assert result.l2_passed

    # ── Layer 3 ────────────────────────────────────────────────────────────────
    _section("LAYER 3 -- Feedback Loop  (Production Trace -> Regression Test)")
    print("  evals/layer3_feedback/  collector.py | analyzer.py | updater.py\n")

    # Good trace already logged by runner.run(); add a simulated bad one
    bad_collector = runner.collector
    bad_response  = faq_agent_bad(golden.question, result.context)
    bad_trace = ProductionTrace(
        trace_id        = "trace-002",
        question        = "Can I return my laptop I bought 60 days ago?",
        context         = result.context,
        agent_response  = bad_response,
        feedback_signal = FeedbackSignal.NEGATIVE,
        llm_judge_score = 0.42,
    )
    bad_collector.log(bad_trace)

    print("  Step 1  Log production traces")
    print("  " + "-" * 55)
    stats = bad_collector.stats()
    print(f"  [OK] trace-faq-reg-001  signal=POSITIVE   judge={result.judge_report.weighted_average:.3f}  -> logged")
    print(f"  [!!] trace-002          signal=NEGATIVE   judge=0.420   -> logged  <- failure candidate")
    print(f"       bad: \"{bad_response[:70]}...\"")
    print(f"\n  Collector stats: {stats}")

    print(f"\n  Step 2  Cluster failure candidates")
    print("  " + "-" * 55)
    from evals.layer3_feedback.analyzer import FailurePatternAnalyzer
    clusters = FailurePatternAnalyzer().cluster_batch(bad_collector.failure_candidates())
    for cluster, info in clusters.items():
        print(f"  [>] {cluster:<26}  count={info['count']}  team={info['team']}")
        print(f"      sample Q: \"{info['sample_question']}\"")

    print(f"\n  Step 3  Promote failure -> golden set YAML")
    print("  " + "-" * 55)
    promoted = GoldenSetUpdater().promote(
        bad_trace,
        correct_answer=(
            "Our return policy covers 30 days from purchase with original receipt. "
            "A 60-day return is outside our policy and cannot be accepted."
        ),
    )
    yaml_out = GoldenSetUpdater.to_yaml_str(promoted)
    print(f"  New entry: {promoted['id']}  ({promoted['category']})\n")
    for line in yaml_out.split("\n"):
        print(f"    {line}")

    _verdict("Layer 3 verdict", True,
             f"Failure caught -> clustered ({bad_trace.failure_cluster}) "
             f"-> promoted as {promoted['id']}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    _section("EVALUATION SUMMARY")
    print()
    m2, m3 = result.metrics[2], result.metrics[3]
    print(f"  [OK]  Layer 1 -- Ground Truth  : gates passed; "
          f"token_f1={m2.score:.3f}  rouge_l={m3.score:.3f}")
    print(f"  [OK]  Layer 2 -- Judgment      : {len(result.code_results)} code evals clean; "
          f"judge {result.judge_report.weighted_average:.3f} >= {JUDGE_THRESHOLD}")
    print(f"  [OK]  Layer 3 -- Feedback Loop : traced, "
          f"clustered ({bad_trace.failure_cluster}), promoted as {promoted['id']}")
    print()
    print(render_token_table(result.token_layers))
    print()
    print(f"  Agent  : faq-bot   Entry : {result.entry_id}   Domain : FAQ Knowledge Base Bot")
    print()


if __name__ == "__main__":
    run_demo()
