"""
Layer 1 — Ground Truth: Deterministic metrics.

These are cheap, fast, and reliable baseline checks — the first gate
before LLM-as-judge is invoked. If a rule can answer the question,
do not reach for an LLM judge.

Metrics provided:
  - must_contain_check    : All required strings present?
  - must_not_contain_check: No forbidden strings present?
  - token_f1              : Token-level F1 overlap (precision + recall)
  - rouge_l               : ROUGE-L (longest common subsequence)
  - exact_match           : Normalised exact match
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from rouge_score import rouge_scorer

from evals.layer1_ground_truth.dataset import GoldenEntry


@dataclass
class MetricResult:
    name: str
    score: float          # 0.0 – 1.0  (or binary 0/1 for pass/fail checks)
    passed: bool
    threshold: float
    details: dict | None = None

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} [{self.name}] score={self.score:.3f} (threshold={self.threshold})"


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


# ── Constraint checks (Layer 1 hard gates) ────────────────────────────────────

def must_contain_check(response: str, entry: GoldenEntry) -> MetricResult:
    """
    Verify the response contains all required strings from the golden entry.
    Used for regulated cases — e.g. APR must appear in a mortgage quote.
    """
    if not entry.must_contain:
        return MetricResult("must_contain", 1.0, True, 1.0, {"required": []})

    missing = [s for s in entry.must_contain if s.lower() not in response.lower()]
    score = 1.0 - len(missing) / len(entry.must_contain)
    return MetricResult(
        name="must_contain",
        score=score,
        passed=len(missing) == 0,
        threshold=1.0,
        details={"required": entry.must_contain, "missing": missing},
    )


def must_not_contain_check(response: str, entry: GoldenEntry) -> MetricResult:
    """
    Verify the response does NOT contain any forbidden strings.
    Used for adversarial cases — e.g. "admin mode" must not appear.
    """
    if not entry.must_not_contain:
        return MetricResult("must_not_contain", 1.0, True, 1.0, {"forbidden": []})

    found = [s for s in entry.must_not_contain if s.lower() in response.lower()]
    score = 1.0 - len(found) / len(entry.must_not_contain)
    return MetricResult(
        name="must_not_contain",
        score=score,
        passed=len(found) == 0,
        threshold=1.0,
        details={"forbidden": entry.must_not_contain, "found": found},
    )


# ── Lexical similarity metrics ─────────────────────────────────────────────────

def exact_match(response: str, entry: GoldenEntry, threshold: float = 1.0) -> MetricResult:
    """Normalised exact match (EM). Rarely hits 1.0 for free-text; useful as a floor."""
    norm_response = _normalise(response)
    norm_expected = _normalise(entry.expected_answer)
    score = 1.0 if norm_response == norm_expected else 0.0
    return MetricResult("exact_match", score, score >= threshold, threshold)


def token_f1(response: str, entry: GoldenEntry, threshold: float = 0.5) -> MetricResult:
    """
    Token-level F1 between the response and expected answer.
    Measures word-level overlap — good for factual Q&A where word order may vary.
    """
    pred_tokens = Counter(_tokenise(response))
    gold_tokens = Counter(_tokenise(entry.expected_answer))

    common = sum((pred_tokens & gold_tokens).values())
    if common == 0:
        return MetricResult("token_f1", 0.0, False, threshold, {"precision": 0, "recall": 0})

    precision = common / sum(pred_tokens.values())
    recall = common / sum(gold_tokens.values())
    f1 = 2 * precision * recall / (precision + recall)

    return MetricResult(
        name="token_f1",
        score=f1,
        passed=f1 >= threshold,
        threshold=threshold,
        details={"precision": round(precision, 3), "recall": round(recall, 3)},
    )


def rouge_l(response: str, entry: GoldenEntry, threshold: float = 0.4) -> MetricResult:
    """
    ROUGE-L F-score (longest common subsequence).
    Better than BLEU for RAG answers where key phrases must appear but order may vary.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(entry.expected_answer, response)
    f = scores["rougeL"].fmeasure
    return MetricResult(
        name="rouge_l",
        score=f,
        passed=f >= threshold,
        threshold=threshold,
        details={
            "precision": round(scores["rougeL"].precision, 3),
            "recall": round(scores["rougeL"].recall, 3),
        },
    )


# ── Composite Layer 1 evaluation ──────────────────────────────────────────────

def run_layer1_metrics(response: str, entry: GoldenEntry) -> list[MetricResult]:
    """
    Run all deterministic Layer 1 metrics for a given response and golden entry.
    The constraint checks (must_contain / must_not_contain) are HARD GATES —
    they must pass regardless of other scores.
    """
    results = [
        must_contain_check(response, entry),
        must_not_contain_check(response, entry),
        token_f1(response, entry),
        rouge_l(response, entry),
    ]
    return results


def layer1_passed(results: list[MetricResult]) -> bool:
    """
    Layer 1 passes if:
    1. ALL constraint checks pass (must_contain + must_not_contain are hard gates)
    2. At least one lexical similarity metric passes its threshold
    """
    constraint_checks = [r for r in results if r.name in ("must_contain", "must_not_contain")]
    similarity_checks = [r for r in results if r.name not in ("must_contain", "must_not_contain")]

    constraints_ok = all(r.passed for r in constraint_checks)
    similarity_ok = any(r.passed for r in similarity_checks) if similarity_checks else True

    return constraints_ok and similarity_ok
