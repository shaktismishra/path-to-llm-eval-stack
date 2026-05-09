"""
EvalRunner – config-driven orchestrator for the 3-layer evaluation stack.

Usage:
    from evals.runner import EvalRunner
    from evals.layer1_ground_truth.dataset import GoldenEntry

    runner = EvalRunner()                         # loads eval_config.yaml
    result = runner.run(entry, agent_fn=my_agent)
    print(result.summary())
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from evals.config import load_eval_config, EVAL_CONFIG, RUBRIC_WEIGHTS, JUDGE_THRESHOLD
from evals.layer1_ground_truth.dataset import GoldenEntry
from evals.layer1_ground_truth.metrics import run_layer1_metrics, layer1_passed
from evals.layer2_judgment.code_evaluators.evaluators import run_code_evaluators
from evals.layer2_judgment.llm_judge.judge import get_judge, JudgeReport
from evals.layer2_judgment.prompts.judge_prompts import get_prompt
from evals.layer3_feedback.collector import (
    ProductionFeedbackCollector, ProductionTrace, FeedbackSignal,
)
from evals.layer3_feedback.analyzer import FailurePatternAnalyzer
from evals.layer3_feedback.updater import GoldenSetUpdater
from evals.retrieval import retrieve_context
from evals.token_tracker import count_tokens, LayerTokens, render_token_table


# ── Per-entry result ───────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    entry_id:       str
    question:       str
    context:        str
    response:       str
    l1_passed:      bool
    l2_passed:      bool
    l3_logged:      bool
    metrics:        list
    code_results:   list
    judge_report:   JudgeReport
    token_layers:   list[LayerTokens]
    failure_cluster: str | None = None

    @property
    def all_passed(self) -> bool:
        return self.l1_passed and self.l2_passed

    def summary(self) -> str:
        l1 = "PASS" if self.l1_passed else "FAIL"
        l2 = "PASS" if self.l2_passed else "FAIL"
        l3 = "logged" if self.l3_logged else "skipped"
        return (
            f"  [{l1}] Layer 1  [{l2}] Layer 2  Layer 3: {l3}\n"
            f"  judge={self.judge_report.weighted_average:.3f}"
            + (f"  cluster={self.failure_cluster}" if self.failure_cluster else "")
        )


# ── Runner ─────────────────────────────────────────────────────────────────────

class EvalRunner:
    """
    Config-driven evaluation runner.

    Args:
        config_path: path to eval_config.yaml (defaults to project root).
        persist:     override the feedback persist flag from config.
    """

    def __init__(self, config_path: Path | str = EVAL_CONFIG,
                 persist: bool | None = None):
        self._root = Path(__file__).parent.parent
        self.cfg   = load_eval_config(Path(config_path))
        _persist   = persist if persist is not None else self.cfg.get(
            "feedback", {}
        ).get("persist", True)
        log_path   = self._root / self.cfg.get("feedback", {}).get(
            "log_path", "evals/layer3_feedback/storage/feedback_log.jsonl"
        )
        self.collector = ProductionFeedbackCollector(persist=_persist, log_path=log_path)
        self.analyzer  = FailurePatternAnalyzer()
        self.updater   = GoldenSetUpdater()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, entry: GoldenEntry,
            agent_fn: Callable[[str, str], str]) -> EvalResult:
        """Run one golden entry through all three evaluation layers."""

        # ── Retrieval ──────────────────────────────────────────────────────────
        ret_cfg  = self.cfg.get("retrieval", {})
        faq_path = self._root / ret_cfg.get("faq_path", "docs/faq.md")
        top_k    = ret_cfg.get("top_k", 2)
        mode     = ret_cfg.get("mode", "keyword")
        s_raw    = ret_cfg.get("structured_path")
        structured_path = (self._root / s_raw) if s_raw else None
        context  = retrieve_context(
            entry.question, faq_path, top_k,
            mode=mode, structured_path=structured_path
        ) or entry.context

        # ── Agent response ─────────────────────────────────────────────────────
        response = agent_fn(entry.question, context)

        # ── Layer 1 ────────────────────────────────────────────────────────────
        l1_in   = count_tokens(entry.question + context + entry.expected_answer + response)
        metrics = run_layer1_metrics(response, entry)
        l1_out  = count_tokens(" ".join(m.details for m in metrics))
        tok_l1  = LayerTokens("Layer 1 - Ground Truth", l1_in, l1_out, llm_call=False)
        l1      = layer1_passed(metrics)

        # ── Layer 2a ───────────────────────────────────────────────────────────
        code_names  = self.cfg.get("evaluators", {}).get("code", [])
        l2a_in      = count_tokens(response) * max(len(code_names), 1)
        code_results= run_code_evaluators(response, code_names)
        l2a_out     = count_tokens(" ".join(r.reason for r in code_results))
        tok_l2a     = LayerTokens("Layer 2a - Code Evals", l2a_in, l2a_out, llm_call=False)
        l2a         = all(r.passed for r in code_results)

        # ── Layer 2b ───────────────────────────────────────────────────────────
        judge_cfg  = self.cfg.get("evaluators", {}).get("judge", {})
        dims       = judge_cfg.get("dimensions", list(RUBRIC_WEIGHTS))
        use_mock   = judge_cfg.get("use_mock", None)   # None → auto-detect
        l2b_in     = sum(
            count_tokens(get_prompt(d, question=entry.question, context=context,
                                    response=response, expected=entry.expected_answer))
            for d in dims
        )
        l2b_out    = 60 * len(dims)                    # ~60 output tokens per dimension
        tok_l2b    = LayerTokens("Layer 2b - LLM Judge", l2b_in, l2b_out, llm_call=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            judge = get_judge(use_mock=use_mock)
        report = judge.evaluate(entry.question, context, response,
                                entry.expected_answer, dimensions=dims)
        l2 = l2a and report.passed

        # ── Layer 3 ────────────────────────────────────────────────────────────
        signal = FeedbackSignal.POSITIVE if l2 else FeedbackSignal.LLM_JUDGE_LOW
        trace  = ProductionTrace(
            trace_id        = f"trace-{entry.id}",
            question        = entry.question,
            context         = context,
            agent_response  = response,
            feedback_signal = signal,
            llm_judge_score = report.weighted_average,
        )
        self.collector.log(trace)
        l3_in  = count_tokens(entry.question + context + response)
        tok_l3 = LayerTokens("Layer 3 - Feedback", l3_in, 0, llm_call=False)

        cluster = None
        if not l2:
            clusters = self.analyzer.cluster_batch([trace])
            cluster  = next(iter(clusters), None)

        return EvalResult(
            entry_id        = entry.id,
            question        = entry.question,
            context         = context,
            response        = response,
            l1_passed       = l1,
            l2_passed       = l2,
            l3_logged       = True,
            metrics         = metrics,
            code_results    = code_results,
            judge_report    = report,
            token_layers    = [tok_l1, tok_l2a, tok_l2b, tok_l3],
            failure_cluster = cluster,
        )

    def triage(self, from_disk: bool = False) -> dict:
        """Cluster all logged failure candidates."""
        return self.analyzer.cluster_batch(
            self.collector.failure_candidates(from_disk=from_disk)
        )

    def promote_failures(
        self,
        correct_answers: dict[str, str],
        write: bool = False,
    ) -> list[dict]:
        """
        Promote failure candidates to golden-set entries.

        Args:
            correct_answers: mapping of trace_id → validated correct answer string.
            write:           When True, persist each promoted entry as a YAML file
                             under golden_set/<category>/<id>.yaml for review.
        Returns list of promoted golden-set entry dicts.
        """
        promoted = []
        for trace in self.collector.failure_candidates():
            answer = correct_answers.get(trace.trace_id)
            if answer:
                entry = self.updater.promote(trace, correct_answer=answer, write=write)
                promoted.append(entry)
        return promoted

    def collector_stats(self) -> dict:
        return self.collector.stats()

    @staticmethod
    def render_token_table(token_layers: list[LayerTokens]) -> str:
        return render_token_table(token_layers)
