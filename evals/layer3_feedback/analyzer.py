"""
Failure pattern analyzer.
Classifies production traces into one of nine root-cause clusters
and routes each cluster to the responsible team.
"""
from __future__ import annotations
import re

# Nine root-cause clusters matched in priority order.
# Each tuple: (cluster_name, regex_pattern_on_question+response)
_CLUSTER_RULES: list[tuple[str, str]] = [
    ("prompt_injection",         r"ignore|forget|admin mode|override|jailbreak"),
    ("policy_exception_missed",  r"final.?sale|no.?return|exception"),
    ("stale_knowledge",          r"\b(2022|2023)\b"),
    ("poor_reasoning",           r"90 days|anytime|no receipt|instant refund|next-day delivery"),
    ("tool_failure",             r"error|timeout|unavailable"),
    ("bad_retrieval",            r"context not found|no results"),
    ("weak_instructions",        r"unclear|ambiguous"),
    ("policy_ambiguity",         r"depends|unclear policy"),
    ("missing_context",          r""),    # catch-all fallthrough
]

_TEAM_ROUTING: dict[str, str] = {
    "prompt_injection":         "Security",
    "policy_exception_missed":  "Compliance",
    "stale_knowledge":          "Content Team",
    "poor_reasoning":           "AI Team",
    "tool_failure":             "Engineering",
    "bad_retrieval":            "Engineering",
    "weak_instructions":        "AI Team",
    "policy_ambiguity":         "Compliance",
    "missing_context":          "Engineering",
}


class FailurePatternAnalyzer:
    """Classifies failure traces into root-cause clusters."""

    def classify(self, trace) -> str:
        text = (trace.question + " " + trace.agent_response).lower()
        for cluster, pattern in _CLUSTER_RULES:
            if pattern and re.search(pattern, text, re.IGNORECASE):
                return cluster
        return "missing_context"

    def owning_team(self, cluster: str) -> str:
        return _TEAM_ROUTING.get(cluster, "Unknown")

    def cluster_batch(self, traces: list) -> dict[str, dict]:
        """
        Classifies a batch of failure candidates.
        Returns a summary dict keyed by cluster name.
        Also sets trace.failure_cluster in-place.
        """
        result: dict[str, list] = {}
        for t in traces:
            cluster = self.classify(t)
            t.failure_cluster = cluster
            result.setdefault(cluster, []).append(t)
        return {
            cluster: {
                "count":           len(items),
                "team":            self.owning_team(cluster),
                "sample_question": items[0].question,
                "trace_ids":       [t.trace_id for t in items],
            }
            for cluster, items in result.items()
        }
