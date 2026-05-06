"""
Layer 3 — Feedback Loop Part 1: Production Trace Collector.

Every week, pull a sample of production traffic weighted toward:
  - Low-confidence outputs
  - Cases the LLM judge flagged as uncertain
  - User escalations and negative feedback
  - New intents the agent hasn't seen before
  - High-risk workflows and tool failures
  - Policy-sensitive responses

The goal: find where the agent is failing BEFORE the same failure
becomes a pattern, not after it costs the business.

A static golden set ages. The world changes. If your golden set doesn't
grow, your eval COVERAGE SHRINKS every week you're in production.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from evals.config import FEEDBACK_LOG_PATH


class FeedbackSignal(str, Enum):
    POSITIVE = "positive"         # User thumbs-up, resolved ticket
    NEGATIVE = "negative"         # User thumbs-down, escalation, complaint
    NEUTRAL = "neutral"           # No signal
    LLM_JUDGE_LOW = "llm_judge_low"   # LLM judge scored below threshold
    TOOL_FAILURE = "tool_failure"      # External tool call failed
    POLICY_FLAG = "policy_flag"        # Compliance rule triggered


@dataclass
class ProductionTrace:
    """
    A single production inference recorded for eval purposes.
    These are the raw inputs to the Layer 3 feedback loop.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    question: str = ""
    context: str = ""             # Retrieved context chunks
    agent_response: str = ""
    llm_judge_score: float | None = None    # Score from Layer 2 LLM judge (if run)
    feedback_signal: FeedbackSignal = FeedbackSignal.NEUTRAL
    user_feedback_text: str = ""  # Verbatim user complaint or praise
    intent: str = ""              # Detected intent / topic (for clustering)
    is_high_risk_flow: bool = False
    is_new_intent: bool = False   # First time agent saw this intent pattern
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    model_version: str = ""       # Agent model version (for regression analysis)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Set after Layer 3 analysis
    failure_cluster: str | None = None
    promoted_to_golden: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feedback_signal"] = self.feedback_signal.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProductionTrace":
        d = dict(d)
        d["feedback_signal"] = FeedbackSignal(d.get("feedback_signal", "neutral"))
        return cls(**d)

    def is_failure_candidate(self, low_confidence_threshold: float = 0.5) -> bool:
        """Is this trace a candidate for failure analysis?"""
        return (
            self.feedback_signal == FeedbackSignal.NEGATIVE
            or self.feedback_signal == FeedbackSignal.TOOL_FAILURE
            or self.feedback_signal == FeedbackSignal.POLICY_FLAG
            or (self.llm_judge_score is not None and self.llm_judge_score < low_confidence_threshold)
        )


class ProductionFeedbackCollector:
    """
    Records production traces to a JSONL feedback log.
    Provides weighted sampling for the Layer 3 analysis pipeline.

    In production, this sits as a middleware layer around your RAG pipeline:
        collector = ProductionFeedbackCollector()
        trace = ProductionTrace(question=q, context=ctx, agent_response=r)
        collector.log(trace)
    """

    def __init__(self, log_path: Path = FEEDBACK_LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()

    def log(self, trace: ProductionTrace) -> str:
        """Append a production trace to the feedback log. Returns trace_id."""
        with self.log_path.open("a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")
        return trace.trace_id

    def sample_for_analysis(
        self,
        n: int = 50,
        weight_failures: bool = True,
        low_confidence_threshold: float = 0.5,
    ) -> list[ProductionTrace]:
        """
        Return a weighted sample of traces for human review / clustering.

        Sampling weights (from the article):
          NEGATIVE feedback     × 4   (highest priority)
          LLM judge low score   × 3
          Tool failures         × 3
          Policy flags          × 3
          New intents           × 2
          High-risk flows       × 2
          Everything else       × 1
        """
        all_traces = self.read_all()

        if not weight_failures:
            import random
            return random.sample(all_traces, min(n, len(all_traces)))

        # Weighted sampling
        import random
        weighted: list[tuple[float, ProductionTrace]] = []
        for trace in all_traces:
            w = 1.0
            if trace.feedback_signal == FeedbackSignal.NEGATIVE:
                w = 4.0
            elif trace.feedback_signal in (FeedbackSignal.TOOL_FAILURE, FeedbackSignal.POLICY_FLAG):
                w = 3.0
            elif trace.llm_judge_score is not None and trace.llm_judge_score < low_confidence_threshold:
                w = 3.0
            elif trace.is_new_intent:
                w = 2.0
            elif trace.is_high_risk_flow:
                w = 2.0
            weighted.append((w, trace))

        population = [t for _, t in weighted]
        weights = [w for w, _ in weighted]
        total = sum(weights)
        probs = [w / total for w in weights]

        chosen = random.choices(population, weights=probs, k=min(n, len(population)))
        return chosen

    def failure_candidates(self, low_confidence_threshold: float = 0.5) -> list[ProductionTrace]:
        """All traces that are candidates for failure analysis."""
        return [t for t in self.read_all() if t.is_failure_candidate(low_confidence_threshold)]

    def read_all(self) -> list[ProductionTrace]:
        traces = []
        with self.log_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(ProductionTrace.from_dict(json.loads(line)))
        return traces

    def stats(self) -> dict:
        traces = self.read_all()
        signal_counts: dict[str, int] = {}
        for t in traces:
            signal_counts[t.feedback_signal.value] = signal_counts.get(t.feedback_signal.value, 0) + 1
        return {
            "total_traces": len(traces),
            "failure_candidates": len(self.failure_candidates()),
            "promoted_to_golden": sum(1 for t in traces if t.promoted_to_golden),
            "by_signal": signal_counts,
        }
