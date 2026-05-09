"""
Golden set updater.
Converts confirmed production failures into versioned YAML entries
that become permanent regression tests in the golden set.

"A failure that doesn't become a regression test will become a
production incident again." — path-to-llm-eval-stack README
"""
from __future__ import annotations
import datetime
from pathlib import Path

from evals.config import GOLDEN_SET

_CATEGORY_MAP: dict[str, str] = {
    "stale_knowledge":        "historical-failures",
    "poor_reasoning":         "historical-failures",
    "missing_context":        "historical-failures",
    "bad_retrieval":          "historical-failures",
    "tool_failure":           "historical-failures",
    "weak_instructions":      "historical-failures",
    "prompt_injection":       "adversarial",
    "policy_exception_missed":"regulated",
    "policy_ambiguity":       "regulated",
}

_JUDGMENT_MAP: dict[str, list[str]] = {
    "prompt_injection":       ["code_evaluator", "eval_no_policy_override"],
    "policy_exception_missed":["code_evaluator", "human_review"],
    "poor_reasoning":         ["code_evaluator", "llm_judge"],
}


class GoldenSetUpdater:
    """Promotes confirmed production failures to golden-set YAML entries."""

    def __init__(self, golden_set_path: Path = GOLDEN_SET):
        self._path     = golden_set_path
        self._promoted: list[dict] = []

    def promote(self, trace, correct_answer: str, write: bool = False) -> dict:
        """
        Build a golden-set entry from a production trace.

        Args:
            trace:          ProductionTrace with failure_cluster set.
            correct_answer: Human-validated correct answer.
            write:          When True, write the entry as a YAML file under
                            golden_set/<category>/<id>.yaml for PR review.
        """
        cluster  = trace.failure_cluster or "missing_context"
        category = _CATEGORY_MAP.get(cluster, "historical-failures")
        idx      = len(self._promoted) + 1
        prefix   = {"historical-failures": "hf",
                    "regulated": "reg",
                    "adversarial": "adv"}[category]
        patterns = _JUDGMENT_MAP.get(cluster, ["code_evaluator", "llm_judge"])
        context_snip = (trace.context[:120] + "...") if len(trace.context) > 120 else trace.context
        entry = {
            "id":               f"{prefix}-{idx:03d}",
            "category":         category,
            "owner":            "faq-team",
            "question":         trace.question,
            "context":          context_snip,
            "expected_answer":  correct_answer,
            "root_cause":       cluster,
            "promoted_from":    trace.trace_id,
            "promoted_at":      datetime.date.today().isoformat(),
            "judgment_patterns": patterns,
        }
        self._promoted.append(entry)
        if write:
            self._write_yaml(entry, category)
        return entry

    def _write_yaml(self, entry: dict, category: str):
        out_dir = self._path / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{entry['id']}.yaml"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(self.to_yaml_str(entry))

    @staticmethod
    def to_yaml_str(entry: dict) -> str:
        lines = ["---"]
        for k, v in entry.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                escaped = str(v).replace("'", "''")
                lines.append(f"{k}: '{escaped}'")
        return "\n".join(lines)
