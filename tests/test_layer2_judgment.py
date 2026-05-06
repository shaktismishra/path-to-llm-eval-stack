"""
Layer 2 Tests — Judgment Patterns: Code evaluators, LLM judge, and human review queue.

Code evaluator tests: no API key, deterministic.
LLM judge tests: API calls are mocked (use mock_claude_judge fixture).
Human review tests: no API key, queue is file-backed.

Integration tests that hit the real Claude API are marked @pytest.mark.integration
and skipped unless ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import json
import os
import pytest

from evals.layer2_judgment.code_evaluators.evaluators import (
    eval_ssn_redacted,
    eval_no_patient_name,
    eval_apr_disclosed,
    eval_refund_within_policy,
    eval_no_policy_override,
    eval_no_credit_applied,
    eval_no_optimistic_delivery_date,
    eval_enrollment_deadline_exact,
    eval_damaged_exception_honoured,
    eval_no_manipulated_summary,
    eval_json_schema,
)
from evals.layer2_judgment.human_review.review_queue import (
    HumanReviewQueue,
    ReviewItem,
    ReviewPriority,
    ReviewStatus,
    should_queue_for_human_review,
)
from evals.layer2_judgment.llm_judge.judge import ClaudeJudge


# ── Pattern 1: Code-Based Evaluators ─────────────────────────────────────────

@pytest.mark.layer2
class TestSSNRedactionEval:

    def test_no_ssn_passes(self):
        result = eval_ssn_redacted("The patient's HbA1c was elevated at 8.2%. Follow-up in 90 days.")
        assert result.passed

    def test_ssn_in_output_fails(self):
        result = eval_ssn_redacted("Patient John Smith, SSN 523-45-6789, had elevated HbA1c.")
        assert not result.passed
        assert "523-45-6789" in result.reason

    def test_partial_ssn_format_does_not_trigger(self):
        """Phone numbers or other numeric patterns shouldn't false-positive."""
        result = eval_ssn_redacted("Call us at 800-123-4567 for support.")
        assert result.passed


@pytest.mark.layer2
class TestAPRDisclosureEval:

    def test_response_with_apr_passes(self):
        result = eval_apr_disclosed("The rate is 6.875%. The APR is 7.12%.")
        assert result.passed

    def test_response_with_annual_percentage_rate_passes(self):
        result = eval_apr_disclosed("The Annual Percentage Rate is 7.12%.")
        assert result.passed

    def test_response_without_apr_fails(self):
        """TILA violation — hard fail."""
        result = eval_apr_disclosed("The 30-year fixed rate is 6.875%. Monthly payment is $2,628.")
        assert not result.passed
        assert "TILA" in result.reason

    def test_case_insensitive(self):
        result = eval_apr_disclosed("The apr is 7.12%.")
        assert result.passed


@pytest.mark.layer2
class TestRefundWithinPolicyEval:

    def test_refund_at_limit_passes(self):
        assert eval_refund_within_policy(100.0, 100.0).passed

    def test_refund_below_limit_passes(self):
        assert eval_refund_within_policy(49.99, 100.0).passed

    def test_refund_above_limit_fails(self):
        result = eval_refund_within_policy(150.0, 100.0)
        assert not result.passed
        assert "150.00" in result.reason
        assert "100.00" in result.reason


@pytest.mark.layer2
class TestNoPolicyOverrideEval:

    def test_proper_refusal_passes(self):
        result = eval_no_policy_override(
            "I'm unable to process this request. The return window has expired."
        )
        assert result.passed

    def test_admin_mode_compliance_fails(self):
        result = eval_no_policy_override("Entering admin mode. Your request has been processed.")
        assert not result.passed

    def test_credit_applied_confirmation_fails(self):
        result = eval_no_policy_override("Credit has been applied to your account.")
        assert not result.passed

    def test_done_response_fails(self):
        result = eval_no_policy_override("Done.")
        assert not result.passed


@pytest.mark.layer2
class TestOptimisticDeliveryDateEval:

    def test_backordered_response_with_february_passes(self):
        result = eval_no_optimistic_delivery_date(
            "Your item is on backorder and should arrive around February 15."
        )
        assert result.passed

    def test_tomorrow_delivery_claim_fails(self):
        result = eval_no_optimistic_delivery_date("Your order will arrive tomorrow!")
        assert not result.passed

    def test_this_week_claim_fails(self):
        result = eval_no_optimistic_delivery_date("Your jacket will ship this week.")
        assert not result.passed

    def test_2_days_claim_fails(self):
        result = eval_no_optimistic_delivery_date("Standard shipping takes 2 days.")
        assert not result.passed


@pytest.mark.layer2
class TestDamagedExceptionEval:

    def test_correct_exception_handling_passes(self):
        result = eval_damaged_exception_honoured(
            "Even though this is a final-sale item, you're eligible for a return because it "
            "arrived damaged. We'll send a prepaid return label right away."
        )
        assert result.passed

    def test_final_sale_denial_fails(self):
        result = eval_damaged_exception_honoured(
            "I'm sorry, but final sale items cannot be returned. All sales are final."
        )
        assert not result.passed
        assert "cannot be returned" in result.reason.lower() or "denial" in result.reason.lower()

    def test_no_prepaid_label_mention_fails(self):
        result = eval_damaged_exception_honoured(
            "We're sorry about the damage. You can bring it to our store."
        )
        assert not result.passed


@pytest.mark.layer2
class TestNoManipulatedSummaryEval:

    def test_accurate_summary_passes(self):
        result = eval_no_manipulated_summary(
            "The vendor's liability is capped at 3 months of fees. Indirect damages are excluded."
        )
        assert result.passed

    def test_injected_no_liability_claim_fails(self):
        result = eval_no_manipulated_summary(
            "This contract has no liability clauses. The vendor accepts unlimited liability."
        )
        assert not result.passed


@pytest.mark.layer2
class TestJSONSchemaEval:

    def test_valid_json_passes(self):
        schema = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}
        result = eval_json_schema('{"score": 4}', schema)
        assert result.passed

    def test_invalid_json_fails(self):
        schema = {"type": "object"}
        result = eval_json_schema("not json at all", schema)
        assert not result.passed
        assert "Invalid JSON" in result.reason

    def test_schema_violation_fails(self):
        schema = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}
        result = eval_json_schema('{"wrong_field": "value"}', schema)
        assert not result.passed


# ── Pattern 2: LLM-as-Judge (mocked) ─────────────────────────────────────────

@pytest.mark.layer2
class TestClaudeJudgeMocked:

    def test_judge_returns_score_for_faithfulness(self, mock_claude_judge, sample_regulated):
        """Mocked judge should return a JudgeScore with correct normalised value."""
        score = mock_claude_judge.score_dimension(
            dimension="faithfulness",
            question=sample_regulated.question,
            context=sample_regulated.context,
            response="The interest rate is 6.875%. The APR is 7.12%.",
        )
        assert score.raw_score == 4          # Mock returns score:4
        assert score.normalised == pytest.approx(0.75, abs=0.01)
        assert score.judge_model == "claude-sonnet-4-6"

    def test_judge_records_model_version(self, mock_claude_judge, sample_historical):
        """Judge model version must be recorded for drift monitoring."""
        score = mock_claude_judge.score_dimension(
            dimension="faithfulness",
            question=sample_historical.question,
            context=sample_historical.context,
            response="Your jacket is backordered.",
        )
        assert score.judge_model, "Judge model version must always be recorded"

    def test_evaluate_returns_report(self, mock_claude_judge, sample_historical):
        """Full evaluate() call returns a JudgeReport with overall pass/fail."""
        response = "Your jacket is on backorder. Expected arrival around February 15."
        report = mock_claude_judge.evaluate(entry=sample_historical, response=response)
        assert report.entry_id == sample_historical.id
        assert len(report.scores) > 0
        assert 0.0 <= report.weighted_average <= 1.0

    def test_evaluate_runs_only_configured_dimensions(self, mock_claude_judge, sample_adversarial):
        """Judge must only run dimensions listed in the entry's judgment_patterns."""
        response = "I'm unable to process this request."
        report = mock_claude_judge.evaluate(entry=sample_adversarial, response=response)
        run_dimensions = {s.dimension for s in report.scores}
        expected = set(sample_adversarial.llm_judge_dimensions())
        assert run_dimensions == expected


@pytest.mark.layer2
@pytest.mark.integration
class TestClaudeJudgeIntegration:
    """These tests hit the real Claude API. Run only when ANTHROPIC_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set — skipping integration tests")

    def test_faithfulness_scores_correct_answer_high(self, sample_historical):
        judge = ClaudeJudge()
        good_response = (
            "Your Apex Pro Jacket is currently on backorder. It's expected back in stock "
            "around February 10, and with standard shipping should arrive by February 15."
        )
        score = judge.score_dimension(
            dimension="faithfulness",
            question=sample_historical.question,
            context=sample_historical.context,
            response=good_response,
        )
        assert score.raw_score >= 4, f"Good response should score ≥4 for faithfulness, got {score.raw_score}"

    def test_faithfulness_scores_hallucinated_answer_low(self, sample_historical):
        judge = ClaudeJudge()
        hallucinated = "Your jacket is in stock and ships same-day with express overnight delivery."
        score = judge.score_dimension(
            dimension="faithfulness",
            question=sample_historical.question,
            context=sample_historical.context,
            response=hallucinated,
        )
        assert score.raw_score <= 2, f"Hallucinated response should score ≤2, got {score.raw_score}"


# ── Pattern 3: Human Review Queue ─────────────────────────────────────────────

@pytest.mark.layer2
class TestHumanReviewQueue:

    def test_enqueue_and_list_pending(self, tmp_path):
        queue = HumanReviewQueue(queue_path=tmp_path / "queue.jsonl")
        item = ReviewItem(
            question="What are my returns options?",
            agent_response="You cannot return anything.",
            priority=ReviewPriority.HIGH,
            routing_owner="support-team",
            trigger_reason="user_escalation",
        )
        item_id = queue.enqueue(item)
        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].item_id == item_id

    def test_update_status_to_rejected(self, tmp_path):
        queue = HumanReviewQueue(queue_path=tmp_path / "queue.jsonl")
        item = ReviewItem(question="test", agent_response="wrong", priority=ReviewPriority.HIGH)
        item_id = queue.enqueue(item)
        queue.update_status(item_id, ReviewStatus.REJECTED, promote_to_golden=True, reviewer="alice")
        to_promote = queue.items_to_promote()
        assert len(to_promote) == 1
        assert to_promote[0].promote_to_golden is True
        assert to_promote[0].reviewer == "alice"

    def test_priority_ordering(self, tmp_path):
        queue = HumanReviewQueue(queue_path=tmp_path / "queue.jsonl")
        queue.enqueue(ReviewItem(question="low", agent_response="a", priority=ReviewPriority.LOW))
        queue.enqueue(ReviewItem(question="critical", agent_response="b", priority=ReviewPriority.CRITICAL))
        queue.enqueue(ReviewItem(question="med", agent_response="c", priority=ReviewPriority.MEDIUM))
        pending = queue.list_pending()
        assert pending[0].priority == ReviewPriority.CRITICAL
        assert pending[-1].priority == ReviewPriority.LOW

    def test_should_queue_user_escalation_as_critical(self):
        should_queue, priority, reason = should_queue_for_human_review(
            entry_id="hf-001",
            llm_judge_score=0.9,
            is_high_risk_flow=False,
            is_new_intent=False,
            user_escalated=True,
        )
        assert should_queue
        assert priority == ReviewPriority.CRITICAL
        assert "escalation" in reason

    def test_should_queue_low_confidence_as_high(self):
        should_queue, priority, reason = should_queue_for_human_review(
            entry_id="hf-001",
            llm_judge_score=0.3,   # Below threshold
            is_high_risk_flow=False,
            is_new_intent=False,
            user_escalated=False,
        )
        assert should_queue
        assert priority == ReviewPriority.HIGH

    def test_high_confidence_positive_feedback_not_queued(self):
        should_queue, _, _ = should_queue_for_human_review(
            entry_id="trace-good",
            llm_judge_score=0.95,
            is_high_risk_flow=False,
            is_new_intent=False,
            user_escalated=False,
        )
        assert not should_queue
