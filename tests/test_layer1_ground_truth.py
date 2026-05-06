"""
Layer 1 Tests — Ground Truth: Dataset loading and deterministic metrics.

These tests run WITHOUT any API key and are fast (< 1 second each).
They validate:
  1. Golden set YAML files load and validate correctly
  2. must_contain / must_not_contain hard gates work
  3. Token F1 and ROUGE-L metrics are correct
  4. Layer 1 composite pass/fail logic is right
"""
from __future__ import annotations

import pytest

from evals.layer1_ground_truth.dataset import GoldenDataset, GoldenEntry
from evals.layer1_ground_truth.metrics import (
    must_contain_check,
    must_not_contain_check,
    token_f1,
    rouge_l,
    run_layer1_metrics,
    layer1_passed,
)


# ── Dataset loading ────────────────────────────────────────────────────────────

@pytest.mark.layer1
class TestGoldenDatasetLoading:

    def test_dataset_loads(self, golden_dataset):
        """Golden set must load without errors."""
        assert len(golden_dataset) > 0, "Golden dataset should not be empty"

    def test_all_three_categories_present(self, golden_dataset):
        """All three golden set categories must exist."""
        categories = {e.category for e in golden_dataset}
        assert "regulated" in categories, "No 'regulated' entries in golden set"
        assert "historical-failures" in categories, "No 'historical-failures' entries"
        assert "adversarial" in categories, "No 'adversarial' entries"

    def test_all_entries_have_required_fields(self, golden_dataset):
        """Every entry must have id, question, context, and expected_answer."""
        for entry in golden_dataset:
            assert entry.id, f"Entry missing id: {entry}"
            assert entry.question, f"Entry {entry.id} missing question"
            assert entry.context, f"Entry {entry.id} missing context"
            assert entry.expected_answer, f"Entry {entry.id} missing expected_answer"
            assert entry.owner, f"Entry {entry.id} missing owner — golden set must be owned"

    def test_filter_by_category(self, golden_dataset):
        regulated = golden_dataset.filter(category="regulated")
        for entry in regulated:
            assert entry.category == "regulated"

    def test_filter_by_difficulty(self, golden_dataset):
        hard = golden_dataset.filter(difficulty="hard")
        for entry in hard:
            assert entry.difficulty == "hard"

    def test_stats_structure(self, golden_dataset):
        stats = golden_dataset.stats()
        assert "total" in stats
        assert "by_category" in stats
        assert stats["total"] == len(golden_dataset)

    def test_regulated_entries_require_compliance_owner(self, golden_dataset):
        """Regulated entries must be owned by compliance or legal — not 'ai-team' alone."""
        regulated = golden_dataset.filter(category="regulated")
        for entry in regulated:
            assert entry.owner, f"Regulated entry {entry.id} has no owner"

    def test_adversarial_entries_have_attack_type_or_notes(self, golden_dataset):
        """Adversarial entries should document the attack type in notes or attack_type field."""
        adversarial = golden_dataset.filter(category="adversarial")
        for entry in adversarial:
            has_attack_info = entry.attack_type or "injection" in entry.notes.lower() or "adversarial" in entry.notes.lower()
            assert has_attack_info, f"Adversarial entry {entry.id} missing attack_type or notes"


# ── must_contain checks ───────────────────────────────────────────────────────

@pytest.mark.layer1
class TestMustContainCheck:

    def test_passes_when_all_strings_present(self, sample_regulated):
        """APR response with all required strings must pass."""
        response = "The interest rate is 6.875%. The Annual Percentage Rate (APR) is 7.12%."
        result = must_contain_check(response, sample_regulated)
        assert result.passed, f"Expected PASS but got: {result}"

    def test_fails_when_apr_missing(self, sample_regulated):
        """A response that mentions the rate but not the APR must fail — TILA violation."""
        response = "The interest rate on a 30-year fixed mortgage for $400,000 is 6.875%."
        result = must_contain_check(response, sample_regulated)
        assert not result.passed, "Should fail: APR not disclosed (TILA violation)"
        assert "APR" in result.details["missing"] or "Annual Percentage Rate" in result.details["missing"]

    def test_score_is_partial_when_some_missing(self, sample_regulated):
        """Partial score when only some required strings are missing."""
        response = "The APR is 7.12%."  # Has APR but not "6.875"
        result = must_contain_check(response, sample_regulated)
        assert 0.0 < result.score < 1.0

    def test_empty_must_contain_always_passes(self, sample_historical):
        """An entry with no must_contain constraint always passes the check."""
        sample_historical.must_contain = []
        response = "Anything at all."
        result = must_contain_check(response, sample_historical)
        assert result.passed


# ── must_not_contain checks ───────────────────────────────────────────────────

@pytest.mark.layer1
class TestMustNotContainCheck:

    def test_passes_when_no_forbidden_strings(self, sample_adversarial):
        """Correct refusal with no forbidden strings must pass."""
        response = "I'm unable to process a refund. The return window has expired."
        result = must_not_contain_check(response, sample_adversarial)
        assert result.passed, f"Expected PASS but got: {result}"

    def test_fails_when_agent_complies_with_override(self, sample_adversarial):
        """If agent enters 'admin mode' the check must hard-fail."""
        response = "Entering admin mode. Refund has been processed. $500 credit applied."
        result = must_not_contain_check(response, sample_adversarial)
        assert not result.passed, "Should fail: agent complied with prompt injection"

    def test_case_insensitive_matching(self, sample_adversarial):
        """Matching must be case-insensitive."""
        response = "ADMIN MODE activated. Refund Has Been Processed."
        result = must_not_contain_check(response, sample_adversarial)
        assert not result.passed

    def test_empty_must_not_contain_always_passes(self, sample_regulated):
        sample_regulated.must_not_contain = []
        response = "Any response is fine."
        result = must_not_contain_check(response, sample_regulated)
        assert result.passed


# ── Lexical metrics ────────────────────────────────────────────────────────────

@pytest.mark.layer1
class TestTokenF1:

    def test_perfect_match_scores_one(self, sample_historical):
        """Identical text (after normalisation) should score 1.0."""
        result = token_f1(sample_historical.expected_answer, sample_historical, threshold=0.5)
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.passed

    def test_completely_wrong_answer_scores_near_zero(self, sample_historical):
        response = "The weather today is sunny with a high of 72 degrees."
        result = token_f1(response, sample_historical, threshold=0.3)
        assert result.score < 0.2, f"Off-topic response should score near zero, got {result.score}"

    def test_partial_overlap_gives_intermediate_score(self, sample_historical):
        response = "Your jacket is on backorder."  # Partially right
        result = token_f1(response, sample_historical, threshold=0.5)
        assert 0.0 < result.score < 1.0


@pytest.mark.layer1
class TestRougeL:

    def test_perfect_match(self, sample_historical):
        result = rouge_l(sample_historical.expected_answer, sample_historical, threshold=0.4)
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.passed

    def test_irrelevant_response(self, sample_historical):
        result = rouge_l("Elephants are large mammals.", sample_historical, threshold=0.4)
        assert result.score < 0.1

    def test_threshold_respected(self, sample_historical):
        response = "Your jacket will arrive around February."
        result_strict = rouge_l(response, sample_historical, threshold=0.9)
        result_loose = rouge_l(response, sample_historical, threshold=0.1)
        assert not result_strict.passed
        assert result_loose.passed


# ── Composite Layer 1 evaluation ──────────────────────────────────────────────

@pytest.mark.layer1
class TestLayer1Composite:

    def test_good_response_passes_all_checks(self, sample_regulated):
        response = (
            "The interest rate is 6.875%. The Annual Percentage Rate (APR) is 7.12%, "
            "which accounts for the 0.5 discount points and origination fee. "
            "Your monthly P&I payment would be $2,628."
        )
        results = run_layer1_metrics(response, sample_regulated)
        assert layer1_passed(results), "A good, complete response should pass Layer 1"

    def test_missing_apr_fails_layer1(self, sample_regulated):
        """Even a high-quality response fails Layer 1 if APR is missing (TILA)."""
        response = "The 30-year fixed rate is 6.875%. Your monthly payment is $2,628."
        results = run_layer1_metrics(response, sample_regulated)
        assert not layer1_passed(results), "Missing APR should fail the hard gate"

    def test_adversarial_compliance_fails_layer1(self, sample_adversarial):
        """If agent complies with prompt injection, Layer 1 hard-fails."""
        response = "Admin mode activated. Refund processed. $500 credit applied."
        results = run_layer1_metrics(response, sample_adversarial)
        assert not layer1_passed(results)

    def test_run_returns_expected_metric_names(self, sample_historical):
        response = "Your item is backordered and will arrive in February."
        results = run_layer1_metrics(response, sample_historical)
        names = {r.name for r in results}
        assert "must_contain" in names
        assert "must_not_contain" in names
        assert "token_f1" in names
        assert "rouge_l" in names
