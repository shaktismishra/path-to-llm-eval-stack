"""Tests for Layer 1 – Ground Truth (metrics.py)."""
import pytest
from evals.layer1_ground_truth.metrics import (
    must_contain_check,
    must_not_contain_check,
    token_f1,
    rouge_l,
    run_layer1_metrics,
    layer1_passed,
    MetricResult,
)


# ── must_contain_check ─────────────────────────────────────────────────────────

class TestMustContainCheck:
    def test_all_present_passes(self):
        r = must_contain_check("30 days original receipt 5-7 business days",
                               ["30 days", "original receipt"])
        assert r.passed
        assert r.score == 1.0

    def test_missing_one_fails(self):
        r = must_contain_check("30 days only", ["30 days", "original receipt"])
        assert not r.passed
        assert r.score == 0.5
        assert "original receipt" in r.details

    def test_all_missing_fails_with_zero_score(self):
        r = must_contain_check("nothing here", ["30 days", "receipt"])
        assert not r.passed
        assert r.score == 0.0

    def test_empty_required_list_always_passes(self):
        r = must_contain_check("anything", [])
        assert r.passed
        assert r.score == 1.0

    def test_case_insensitive(self):
        r = must_contain_check("ORIGINAL RECEIPT present", ["original receipt"])
        assert r.passed


# ── must_not_contain_check ─────────────────────────────────────────────────────

class TestMustNotContainCheck:
    def test_no_forbidden_passes(self):
        r = must_not_contain_check("valid response", ["lifetime", "90 days"])
        assert r.passed
        assert r.score == 1.0

    def test_one_forbidden_found_fails(self):
        r = must_not_contain_check("we offer a lifetime guarantee",
                                   ["lifetime", "90 days"])
        assert not r.passed
        assert r.score == 0.5

    def test_all_forbidden_found_zero_score(self):
        r = must_not_contain_check("lifetime 90 days", ["lifetime", "90 days"])
        assert not r.passed
        assert r.score == 0.0

    def test_empty_forbidden_list_always_passes(self):
        r = must_not_contain_check("any response", [])
        assert r.passed

    def test_case_insensitive(self):
        r = must_not_contain_check("LIFETIME warranty", ["lifetime"])
        assert not r.passed


# ── token_f1 ───────────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_identical_texts_score_one(self):
        text = "return within 30 days with original receipt"
        r = token_f1(text, text)
        assert r.score == pytest.approx(1.0)
        assert r.passed

    def test_no_overlap_score_zero(self):
        r = token_f1("completely different words", "other unrelated content")
        assert r.score == 0.0
        assert not r.passed

    def test_partial_overlap_between_zero_and_one(self):
        r = token_f1("return 30 days receipt packaging",
                     "30 days original receipt and packaging refunds")
        assert 0.0 < r.score < 1.0

    def test_passes_above_threshold(self):
        text = "30 days original receipt packaging refunds business days"
        r = token_f1(text, text, threshold=0.40)
        assert r.passed

    def test_fails_below_threshold(self):
        r = token_f1("one word", "completely different long sentence with many words",
                     threshold=0.40)
        assert not r.passed


# ── rouge_l ────────────────────────────────────────────────────────────────────

class TestRougeL:
    def test_identical_texts_score_one(self):
        text = "return within 30 days with original receipt"
        r = rouge_l(text, text)
        assert r.score == pytest.approx(1.0)
        assert r.passed

    def test_no_overlap_score_zero(self):
        r = rouge_l("apple banana cherry", "dog cat elephant")
        assert r.score == 0.0
        assert not r.passed

    def test_partial_overlap_passes_threshold(self, good_response, good_entry):
        r = rouge_l(good_response, good_entry.expected_answer, threshold=0.35)
        assert r.passed

    def test_lcs_detail_present(self):
        r = rouge_l("return 30 days receipt", "30 days receipt policy")
        assert "LCS=" in r.details


# ── layer1_passed ──────────────────────────────────────────────────────────────

class TestLayer1Passed:
    def _make(self, name, passed, score=1.0):
        return MetricResult(name=name, score=score, passed=passed, threshold=0.5)

    def test_all_pass_returns_true(self):
        metrics = [
            self._make("must_contain", True),
            self._make("must_not_contain", True),
            self._make("token_f1", True, 0.6),
            self._make("rouge_l", True, 0.5),
        ]
        assert layer1_passed(metrics)

    def test_constraint_gate_fails_blocks_pass(self):
        metrics = [
            self._make("must_contain", False),          # gate fails
            self._make("must_not_contain", True),
            self._make("token_f1", True, 0.6),
            self._make("rouge_l", True, 0.5),
        ]
        assert not layer1_passed(metrics)

    def test_at_least_one_similarity_required(self):
        metrics = [
            self._make("must_contain", True),
            self._make("must_not_contain", True),
            self._make("token_f1", False, 0.1),         # both similarity fail
            self._make("rouge_l", False, 0.1),
        ]
        assert not layer1_passed(metrics)

    def test_one_similarity_sufficient(self):
        metrics = [
            self._make("must_contain", True),
            self._make("must_not_contain", True),
            self._make("token_f1", False, 0.1),
            self._make("rouge_l", True, 0.5),            # only rouge passes
        ]
        assert layer1_passed(metrics)

    def test_full_pipeline_on_good_response(self, good_entry, good_response):
        metrics = run_layer1_metrics(good_response, good_entry)
        assert layer1_passed(metrics)
