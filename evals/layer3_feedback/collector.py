"""
Production trace collector.
Logs every inference as a ProductionTrace; identifies failure candidates
for weekly triage so the golden set never goes stale.

Persistence: traces are appended to a JSONL file (one JSON object per line).
The in-memory list supports the demo runner without requiring disk I/O.
"""
from __future__ import annotations
import datetime
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from evals.config import FEEDBACK_LOG


class FeedbackSignal(Enum):
    POSITIVE      = "positive"       # user approval / resolved ticket
    NEGATIVE      = "negative"       # user disapproval / escalation
    NEUTRAL       = "neutral"
    LLM_JUDGE_LOW = "llm_judge_low"  # below-threshold judge score
    TOOL_FAILURE  = "tool_failure"
    POLICY_FLAG   = "policy_flag"    # compliance rule violation


@dataclass
class ProductionTrace:
    trace_id: str
    question: str
    context: str
    agent_response: str
    feedback_signal: FeedbackSignal
    llm_judge_score: Optional[float] = None
    failure_cluster: Optional[str]   = None
    promoted: bool                   = False
    timestamp: str = field(
        default_factory=lambda: (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")
        )
    )

    def is_failure_candidate(self) -> bool:
        return self.feedback_signal in (
            FeedbackSignal.NEGATIVE,
            FeedbackSignal.LLM_JUDGE_LOW,
            FeedbackSignal.TOOL_FAILURE,
            FeedbackSignal.POLICY_FLAG,
        )


def _serialise(trace: ProductionTrace) -> dict:
    """Convert a ProductionTrace to a JSON-serialisable dict."""
    d = asdict(trace)
    d["feedback_signal"] = trace.feedback_signal.value
    return d


class ProductionFeedbackCollector:
    """
    Appends traces to a JSONL file and maintains an in-memory list for
    fast querying within the same process.

    Args:
        persist:  write traces to disk when True (default True).
        log_path: path to the JSONL file (defaults to config value).
    """

    def __init__(self, persist: bool = True, log_path: Path = FEEDBACK_LOG):
        self._persist  = persist
        self._log_path = Path(log_path)
        self._memory:  list[ProductionTrace] = []

    def log(self, trace: ProductionTrace) -> None:
        self._memory.append(trace)
        if self._persist:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_serialise(trace)) + "\n")

    def read_all(self) -> list[ProductionTrace]:
        """Load all persisted traces from the JSONL file."""
        if not self._log_path.exists():
            return []
        traces: list[ProductionTrace] = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["feedback_signal"] = FeedbackSignal(d["feedback_signal"])
                traces.append(ProductionTrace(**d))
        return traces

    def failure_candidates(self, from_disk: bool = False) -> list[ProductionTrace]:
        source = self.read_all() if from_disk else self._memory
        return [t for t in source if t.is_failure_candidate()]

    def stats(self, from_disk: bool = False) -> dict:
        source = self.read_all() if from_disk else self._memory
        return {
            "total_traces":       len(source),
            "failure_candidates": sum(1 for t in source if t.is_failure_candidate()),
            "by_signal":          dict(Counter(t.feedback_signal.value for t in source)),
        }
