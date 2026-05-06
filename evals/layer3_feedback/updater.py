"""
Layer 3 — Feedback Loop Part 3: Golden Set Updater (Compounding Loop).

Every confirmed failure becomes a new ground truth case.
Same week. Versioned. Reviewed. Owned.

The article's pipeline:
  Failure detected Tuesday →
    Clustered and root-caused Wednesday →
      New eval case written Thursday →
        Added to golden set and merged Friday →
          Regression test runs in next deployment cycle

This is the MOAT. Teams that promote production failures into regression
tests compound their eval coverage. The golden set grows sharper every
week the business runs.

A failure that doesn't become a regression test will become a
production incident again.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evals.config import GOLDEN_SET_DIR
from evals.layer3_feedback.analyzer import FailureAnalysis, FailureCluster
from evals.layer3_feedback.collector import ProductionTrace


# Map failure clusters to golden set categories
CLUSTER_TO_CATEGORY: dict[FailureCluster, str] = {
    FailureCluster.MISSING_CONTEXT: "historical-failures",
    FailureCluster.BAD_RETRIEVAL: "historical-failures",
    FailureCluster.WEAK_INSTRUCTIONS: "historical-failures",
    FailureCluster.TOOL_FAILURE: "historical-failures",
    FailureCluster.POLICY_AMBIGUITY: "regulated",
    FailureCluster.POOR_REASONING: "historical-failures",
    FailureCluster.STALE_KNOWLEDGE: "historical-failures",
    FailureCluster.POLICY_EXCEPTION_MISSED: "historical-failures",
    FailureCluster.PROMPT_INJECTION: "adversarial",
    FailureCluster.UNKNOWN: "historical-failures",
}


class GoldenSetUpdater:
    """
    Promotes confirmed production failures into the governed golden set.

    This is the compounding part of the eval stack. Each confirmed failure
    becomes a permanent regression test, so the same mistake cannot ship
    again without triggering a CI failure.

    Usage:
        trace = ...         # Production trace that failed
        analysis = ...      # FailureAnalysis from the Analyzer
        updater = GoldenSetUpdater()
        path = updater.promote(trace, analysis, correct_answer="...", owner="support-team")
        print(f"New golden entry written: {path}")
    """

    def __init__(self, golden_set_dir: Path = GOLDEN_SET_DIR) -> None:
        self.golden_set_dir = golden_set_dir

    def promote(
        self,
        trace: ProductionTrace,
        analysis: FailureAnalysis,
        correct_answer: str,
        owner: str,
        must_contain: list[str] | None = None,
        must_not_contain: list[str] | None = None,
        notes: str = "",
    ) -> Path:
        """
        Convert a confirmed production failure into a new golden set entry (YAML).

        The YAML file is written to the appropriate sub-directory and is ready
        to be committed via a pull request. PR review IS the human sign-off.

        Returns the path to the newly written YAML file.
        """
        category = CLUSTER_TO_CATEGORY[analysis.cluster]
        entry_id = self._next_id(category)
        slug = self._slugify(trace.question)[:50]
        filename = f"{entry_id}-{slug}.yaml"
        output_path = self.golden_set_dir / category / filename

        judgment_patterns = self._default_judgment_patterns(analysis.cluster)
        if must_contain:
            judgment_patterns.append({"code_evaluator": "eval_must_contain_custom"})
        if must_not_contain:
            judgment_patterns.append({"code_evaluator": "eval_must_not_contain_custom"})

        entry = {
            "id": entry_id,
            "category": category,
            "sub_category": analysis.cluster.value,
            "difficulty": "medium",
            "source": "promoted_from_production",
            "owner": owner,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "question": trace.question,
            "context": trace.context,
            "expected_answer": correct_answer,
            "must_contain": must_contain or [],
            "must_not_contain": must_not_contain or [],
            "judgment_patterns": judgment_patterns,
            "failure_root_cause": analysis.cluster.value,
            "failure_description": analysis.description,
            "incident_cost": f"Production trace {trace.trace_id} — feedback: {trace.feedback_signal.value}",
            "regression_added": datetime.now(timezone.utc).date().isoformat(),
            "original_trace_id": trace.trace_id,
            "notes": notes or (
                f"Promoted from production failure. Root cause: {analysis.cluster.value}. "
                f"Recommended action: {analysis.recommended_action}"
            ),
        }

        with output_path.open("w") as f:
            yaml.dump(entry, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return output_path

    def _next_id(self, category: str) -> str:
        """Generate the next sequential ID for a category (e.g. hf-004, adv-003)."""
        prefix_map = {
            "regulated": "reg",
            "historical-failures": "hf",
            "adversarial": "adv",
        }
        prefix = prefix_map.get(category, "unk")
        existing = list((self.golden_set_dir / category).glob("*.yaml"))
        # Find the highest existing number
        numbers = []
        for p in existing:
            m = re.search(rf"{prefix}-(\d+)", p.stem)
            if m:
                numbers.append(int(m.group(1)))
        next_num = (max(numbers) + 1) if numbers else 1
        return f"{prefix}-{next_num:03d}"

    def _slugify(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        return text.strip("-")

    def _default_judgment_patterns(self, cluster: FailureCluster) -> list[dict]:
        """Suggest sensible default judgment patterns based on cluster type."""
        if cluster == FailureCluster.PROMPT_INJECTION:
            return [
                {"code_evaluator": "eval_no_policy_override"},
                {"llm_judge": "faithfulness"},
            ]
        elif cluster == FailureCluster.POLICY_EXCEPTION_MISSED:
            return [
                {"code_evaluator": "eval_damaged_exception_honoured"},
                {"llm_judge": "faithfulness"},
                {"llm_judge": "completeness"},
            ]
        elif cluster == FailureCluster.POLICY_AMBIGUITY:
            return [
                {"llm_judge": "faithfulness"},
                {"llm_judge": "correctness"},
                {"human_review": True},
            ]
        elif cluster in (FailureCluster.MISSING_CONTEXT, FailureCluster.STALE_KNOWLEDGE):
            return [
                {"llm_judge": "faithfulness"},
                {"llm_judge": "correctness"},
            ]
        else:
            return [
                {"llm_judge": "faithfulness"},
                {"llm_judge": "relevance"},
            ]
