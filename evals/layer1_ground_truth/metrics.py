"""
Layer 1 deterministic metrics.
These run BEFORE the LLM judge and act as hard gates.

Constraint checks (hard gates):
  must_contain_check     – all required strings must be present
  must_not_contain_check – no forbidden strings may appear

Similarity metrics (at least one must pass):
  token_f1   – word-level F1 overlap
  rouge_l    – longest-common-subsequence F-score
"""
from __future__ import annotations
import string
from collections import Counter
from dataclasses import dataclass

from evals.config import TOKEN_F1_THRESHOLD, ROUGE_L_THRESHOLD


@dataclass
class MetricResult:
    name: str
    score: float        # 0.0 – 1.0
    passed: bool
    threshold: float
    details: str = ""


# ── Text helpers ───────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


# ── Hard gates ─────────────────────────────────────────────────────────────────

def must_contain_check(response: str, required: list[str],
                       threshold: float = 1.0) -> MetricResult:
    missing = [r for r in required if r.lower() not in response.lower()]
    score   = 1.0 - len(missing) / max(len(required), 1)
    passed  = len(missing) == 0
    details = f"Missing: {missing}" if missing else "All required strings present"
    return MetricResult("must_contain", round(score, 3), passed, threshold, details)


def must_not_contain_check(response: str, forbidden: list[str],
                            threshold: float = 1.0) -> MetricResult:
    found   = [f for f in forbidden if f.lower() in response.lower()]
    score   = 1.0 - len(found) / max(len(forbidden), 1)
    passed  = len(found) == 0
    details = f"Forbidden found: {found}" if found else "No forbidden strings found"
    return MetricResult("must_not_contain", round(score, 3), passed, threshold, details)


# ── Similarity metrics ─────────────────────────────────────────────────────────

def token_f1(response: str, expected: str,
             threshold: float = TOKEN_F1_THRESHOLD) -> MetricResult:
    resp_tok = Counter(_tokenise(response))
    exp_tok  = Counter(_tokenise(expected))
    common   = sum((resp_tok & exp_tok).values())
    if common == 0:
        return MetricResult("token_f1", 0.0, False, threshold, "No token overlap")
    precision = common / sum(resp_tok.values())
    recall    = common / sum(exp_tok.values())
    f1 = 2 * precision * recall / (precision + recall)
    return MetricResult("token_f1", round(f1, 3), f1 >= threshold, threshold,
                        f"precision={precision:.2f}  recall={recall:.2f}")


def _lcs(a: list, b: list) -> int:
    """Rolling-array LCS — O(m*n) time, O(n) space."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def rouge_l(response: str, expected: str,
            threshold: float = ROUGE_L_THRESHOLD) -> MetricResult:
    r_tok = _tokenise(response)
    e_tok = _tokenise(expected)
    lcs   = _lcs(r_tok, e_tok)
    if lcs == 0:
        return MetricResult("rouge_l", 0.0, False, threshold, "No common subsequence")
    precision = lcs / len(r_tok)
    recall    = lcs / len(e_tok)
    f = 2 * precision * recall / (precision + recall)
    return MetricResult("rouge_l", round(f, 3), f >= threshold, threshold,
                        f"LCS={lcs} tokens  precision={precision:.2f}  recall={recall:.2f}")


# ── Composite evaluation ───────────────────────────────────────────────────────

def run_layer1_metrics(response: str, entry) -> list[MetricResult]:
    """Run all five checks against a GoldenEntry."""
    return [
        must_contain_check(response, entry.must_contain),
        must_not_contain_check(response, entry.must_not_contain),
        token_f1(response, entry.expected_answer),
        rouge_l(response, entry.expected_answer),
    ]


def layer1_passed(metrics: list[MetricResult]) -> bool:
    """Passes iff ALL constraint gates pass AND at least ONE similarity metric passes."""
    constraints = [m for m in metrics if m.name in ("must_contain", "must_not_contain")]
    similarity  = [m for m in metrics if m.name in ("token_f1", "rouge_l")]
    return all(m.passed for m in constraints) and any(m.passed for m in similarity)
