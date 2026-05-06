"""
Layer 3 — Feedback Loop Part 2: Failure Pattern Analyzer.

Don't treat every failure as a one-off. Group failures by root cause:

  - missing_context    : Agent didn't have the right information
  - bad_retrieval      : Right information existed but wasn't retrieved
  - weak_instructions  : System prompt was ambiguous or incomplete
  - tool_failure       : External call returned stale or wrong data
  - policy_ambiguity   : Business rule was unclear or conflicting
  - poor_reasoning     : Model made a logical error with good inputs
  - stale_knowledge    : Information existed but was outdated
  - policy_exception_missed : Agent applied general rule, missed exception

Once failures are clustered, the team sees the PATTERN instead of
debating anecdotes. Route each cluster to the team that owns the domain:
  - compliance owns policy gaps
  - engineering owns tool failures
  - content owners fix stale knowledge sources
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from evals.layer3_feedback.collector import ProductionTrace, FeedbackSignal


class FailureCluster(str, Enum):
    MISSING_CONTEXT = "missing_context"
    BAD_RETRIEVAL = "bad_retrieval"
    WEAK_INSTRUCTIONS = "weak_instructions"
    TOOL_FAILURE = "tool_failure"
    POLICY_AMBIGUITY = "policy_ambiguity"
    POOR_REASONING = "poor_reasoning"
    STALE_KNOWLEDGE = "stale_knowledge"
    POLICY_EXCEPTION_MISSED = "policy_exception_missed"
    PROMPT_INJECTION = "prompt_injection"
    UNKNOWN = "unknown"


# Domain routing — who owns each cluster
CLUSTER_ROUTING: dict[FailureCluster, str] = {
    FailureCluster.MISSING_CONTEXT: "content-team",
    FailureCluster.BAD_RETRIEVAL: "engineering",
    FailureCluster.WEAK_INSTRUCTIONS: "ai-team",
    FailureCluster.TOOL_FAILURE: "engineering",
    FailureCluster.POLICY_AMBIGUITY: "compliance",
    FailureCluster.POOR_REASONING: "ai-team",
    FailureCluster.STALE_KNOWLEDGE: "content-team",
    FailureCluster.POLICY_EXCEPTION_MISSED: "compliance",
    FailureCluster.PROMPT_INJECTION: "security",
    FailureCluster.UNKNOWN: "ai-team",
}


@dataclass
class FailureAnalysis:
    """Result of clustering a single production trace failure."""
    trace_id: str
    cluster: FailureCluster
    confidence: float
    routing_owner: str
    description: str
    recommended_action: str


@dataclass
class ClusterSummary:
    """Aggregated view of a failure cluster across multiple traces."""
    cluster: FailureCluster
    count: int
    traces: list[str] = field(default_factory=list)
    routing_owner: str = ""
    example_questions: list[str] = field(default_factory=list)


class FailurePatternAnalyzer:
    """
    Rule-based failure cluster classifier.

    In a mature system, replace or augment the rule-based classifier with
    an LLM-based classifier (using Claude) that reads the full trace and
    returns a structured cluster + reasoning.
    """

    def classify(self, trace: ProductionTrace) -> FailureAnalysis:
        cluster, confidence, description = self._rule_based_classify(trace)
        routing_owner = CLUSTER_ROUTING[cluster]
        action = self._recommended_action(cluster)
        return FailureAnalysis(
            trace_id=trace.trace_id,
            cluster=cluster,
            confidence=confidence,
            routing_owner=routing_owner,
            description=description,
            recommended_action=action,
        )

    def cluster_batch(self, traces: list[ProductionTrace]) -> list[ClusterSummary]:
        analyses = [self.classify(t) for t in traces if t.is_failure_candidate()]
        cluster_map: dict[FailureCluster, ClusterSummary] = {}
        for analysis in analyses:
            if analysis.cluster not in cluster_map:
                cluster_map[analysis.cluster] = ClusterSummary(
                    cluster=analysis.cluster,
                    count=0,
                    routing_owner=analysis.routing_owner,
                )
            summary = cluster_map[analysis.cluster]
            summary.count += 1
            summary.traces.append(analysis.trace_id)
        trace_map = {t.trace_id: t for t in traces}
        for cluster, summary in cluster_map.items():
            summary.example_questions = [
                trace_map[tid].question[:120]
                for tid in summary.traces[:3]
                if tid in trace_map
            ]
        return sorted(cluster_map.values(), key=lambda s: s.count, reverse=True)

    def _rule_based_classify(self, trace: ProductionTrace) -> tuple[FailureCluster, float, str]:
        response = trace.agent_response.lower()
        question = trace.question.lower()
        context = trace.context.lower()

        # Tool failure
        if trace.feedback_signal == FeedbackSignal.TOOL_FAILURE or any(
            tc.get("status") == "error" for tc in trace.tool_calls
        ):
            return (FailureCluster.TOOL_FAILURE, 0.9,
                    "An external tool call failed or returned an error status.")

        # Prompt injection — check BEFORE generic policy flag
        if any(kw in question for kw in ["forget", "ignore", "admin mode", "new instructions"]):
            return (FailureCluster.PROMPT_INJECTION, 0.85,
                    "Question contains prompt injection patterns (forget / admin mode / override).")

        # Context emptiness — bad retrieval
        if len(trace.context.strip()) < 20:
            return (FailureCluster.BAD_RETRIEVAL, 0.8,
                    "Context is nearly empty — retrieval likely failed to surface relevant chunks.")

        # Policy flag (non-injection)
        if trace.feedback_signal == FeedbackSignal.POLICY_FLAG:
            return (FailureCluster.POLICY_AMBIGUITY, 0.75,
                    "A compliance rule was flagged. Policy may be ambiguous or incomplete.")

        # Policy exception missed
        if "exception" in context and (
            "cannot" in response or "not eligible" in response or "all sales" in response
        ):
            return (FailureCluster.POLICY_EXCEPTION_MISSED, 0.75,
                    "Context contains an exception clause, but the response applied the general rule instead.")

        # Backorder miss
        if "backorder" in context and any(
            kw in response for kw in ["tomorrow", "today", "2 days", "next day"]
        ):
            return (FailureCluster.MISSING_CONTEXT, 0.8,
                    "Context indicates backordered status, but agent gave an optimistic delivery estimate.")

        # Stale data
        if "prior year" in context and ("2022" in response or "2021" in response):
            return (FailureCluster.STALE_KNOWLEDGE, 0.7,
                    "Response appears to use prior-year data when current-year data was in context.")

        return (FailureCluster.POOR_REASONING, 0.4,
                "Could not identify a specific structural cause — likely a model reasoning error.")

    def _recommended_action(self, cluster: FailureCluster) -> str:
        actions = {
            FailureCluster.MISSING_CONTEXT: "Add the missing information to the knowledge base and re-index.",
            FailureCluster.BAD_RETRIEVAL: "Audit retrieval pipeline — check chunk size, embedding model, and similarity threshold.",
            FailureCluster.WEAK_INSTRUCTIONS: "Revise system prompt to add explicit handling for this case.",
            FailureCluster.TOOL_FAILURE: "Review the failing tool API; add fallback handling and alerting.",
            FailureCluster.POLICY_AMBIGUITY: "Work with compliance to clarify the policy and update the knowledge base.",
            FailureCluster.POOR_REASONING: "Add this as a few-shot example in the system prompt or fine-tuning set.",
            FailureCluster.STALE_KNOWLEDGE: "Update the source document and re-ingest. Add date-sensitivity checks.",
            FailureCluster.POLICY_EXCEPTION_MISSED: "Add the exception clause as a separate, prominently indexed chunk.",
            FailureCluster.PROMPT_INJECTION: "Review system prompt injection defences; add this variant to adversarial golden set.",
            FailureCluster.UNKNOWN: "Manual review required — classify and route to appropriate team.",
        }
        return actions.get(cluster, "Manual review required.")
