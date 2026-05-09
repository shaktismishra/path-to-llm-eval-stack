"""Shared pytest fixtures."""
import pytest
from evals.layer1_ground_truth.dataset import GoldenEntry
from evals.layer3_feedback.collector import ProductionTrace, FeedbackSignal


@pytest.fixture
def good_entry() -> GoldenEntry:
    return GoldenEntry(
        id="test-reg-001",
        category="regulated",
        question="What is your return policy for electronics?",
        context=(
            "Our return policy allows returns within 30 days of purchase with "
            "the original receipt and original packaging. Refunds are processed "
            "within 5-7 business days to the original payment method."
        ),
        expected_answer=(
            "Electronics can be returned within 30 days with original receipt "
            "and packaging. Refunds take 5-7 business days."
        ),
        must_contain=["30 days", "original receipt", "5-7 business days"],
        must_not_contain=["lifetime", "90 days", "immediate refund"],
        judgment_patterns=["code_evaluator", "llm_judge"],
    )


@pytest.fixture
def good_response() -> str:
    return (
        "You can return electronics within 30 days of purchase. "
        "Please bring your original receipt and ensure the item is in its "
        "original packaging. Your refund will be processed within 5-7 business "
        "days back to your original payment method."
    )


@pytest.fixture
def bad_response() -> str:
    return (
        "We offer next-day delivery on all electronics! "
        "You can return items anytime within 90 days -- no receipt needed. "
        "Refunds are instant."
    )


@pytest.fixture
def good_trace(good_entry, good_response) -> ProductionTrace:
    return ProductionTrace(
        trace_id="trace-001",
        question=good_entry.question,
        context=good_entry.context,
        agent_response=good_response,
        feedback_signal=FeedbackSignal.POSITIVE,
        llm_judge_score=0.81,
    )


@pytest.fixture
def bad_trace(good_entry, bad_response) -> ProductionTrace:
    return ProductionTrace(
        trace_id="trace-002",
        question="Can I return my laptop I bought 60 days ago?",
        context=good_entry.context,
        agent_response=bad_response,
        feedback_signal=FeedbackSignal.NEGATIVE,
        llm_judge_score=0.42,
    )
