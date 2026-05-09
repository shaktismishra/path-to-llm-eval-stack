"""
Pattern 2 – LLM-as-Judge.

Production : ClaudeJudge   – calls Anthropic API (requires ANTHROPIC_API_KEY).
Development: MockClaudeJudge – deterministic scores, no API call.

Use get_judge() to get the right implementation automatically:
  - Explicit USE_MOCK_JUDGE=true  → MockClaudeJudge
  - ANTHROPIC_API_KEY set         → ClaudeJudge
  - Neither                       → MockClaudeJudge (with a warning)

Rubric weights: Faithfulness 0.35 · Relevance 0.25 · Completeness 0.20 · Correctness 0.20
Score scale: raw 1–5  →  normalised (raw-1)/4  →  0.0–1.0
Pass threshold: JUDGE_THRESHOLD (default 0.70, env-overridable)
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass

from evals.config import RUBRIC_WEIGHTS, JUDGE_THRESHOLD, JUDGE_MODEL


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    dimension: str
    raw_score: int        # 1–5 rubric scale
    normalized: float     # (raw-1)/4 → 0.0–1.0
    threshold: float
    passed: bool
    reasoning: str
    judge_model: str = JUDGE_MODEL


@dataclass
class JudgeReport:
    scores: list[JudgeScore]
    weighted_average: float
    passed: bool
    judge_model: str = JUDGE_MODEL

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"  Judge model     : {self.judge_model}",
            f"  Weighted score  : {self.weighted_average:.3f}"
            f"  (threshold {JUDGE_THRESHOLD})  ->  {verdict}",
            "",
        ]
        for s in self.scores:
            icon    = "OK" if s.passed else "!!"
            weight  = RUBRIC_WEIGHTS.get(s.dimension, 0)
            contrib = weight * s.normalized
            lines.append(
                f"  [{icon}] {s.dimension:<14}"
                f"  score={s.normalized:.2f}  (raw {s.raw_score}/5)"
                f"  weight={weight}  contrib={contrib:.3f}"
            )
            lines.append(f"      {s.reasoning}")
        return "\n".join(lines)

    @property
    def failed_dimensions(self) -> list[str]:
        return [s.dimension for s in self.scores if not s.passed]


# ── Mock judge (offline / CI) ──────────────────────────────────────────────────

class MockClaudeJudge:
    """
    Deterministic stand-in – no API call.
    Scores are calibrated to a solid-but-not-perfect FAQ response.
    Swap with ClaudeJudge (or let get_judge() choose) for production.
    """

    _MOCK: dict[str, tuple[int, str]] = {
        "faithfulness": (4, "Grounded in context; no unsupported claims detected."),
        "relevance":    (5, "Directly and completely addresses the return-policy question."),
        "completeness": (4, "Covers timeframe, receipt, packaging, and refund timeline."),
        "correctness":  (4, "Matches expected answer on all key facts."),
    }

    def evaluate(self, question: str, context: str, response: str,
                 expected: str, dimensions: list[str] | None = None) -> JudgeReport:
        dims = dimensions or list(RUBRIC_WEIGHTS)
        scores, weighted_sum = [], 0.0
        for dim in dims:
            weight       = RUBRIC_WEIGHTS.get(dim, 0)
            raw, reason  = self._MOCK.get(dim, (3, "No mock score configured."))
            norm         = (raw - 1) / 4
            scores.append(JudgeScore(dim, raw, round(norm, 3),
                                     JUDGE_THRESHOLD, norm >= JUDGE_THRESHOLD,
                                     reason, judge_model="mock-" + JUDGE_MODEL))
            weighted_sum += weight * norm
        return JudgeReport(scores, round(weighted_sum, 3),
                           weighted_sum >= JUDGE_THRESHOLD,
                           judge_model="mock-" + JUDGE_MODEL)


# ── Real Claude judge ──────────────────────────────────────────────────────────

class ClaudeJudge:
    """
    Production LLM judge – calls the Anthropic Messages API.
    Requires: pip install anthropic  and  ANTHROPIC_API_KEY env var.

    Retries up to MAX_RETRIES times with exponential back-off on transient errors.
    Locks judge_model in JudgeScore so score drift from model updates is traceable.
    """

    MAX_RETRIES  = 3
    BACKOFF_BASE = 2.0   # seconds

    def __init__(self, api_key: str | None = None, model: str = JUDGE_MODEL):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. Run:  pip install anthropic"
            ) from exc
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "")
        )
        self.model = model

    def _call_with_retry(self, prompt: str) -> dict:
        """Call the API and parse JSON, retrying on transient failures."""
        import anthropic
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text
                text = re.sub(r"```(?:json)?|```", "", text).strip()
                return json.loads(text)
            except (anthropic.APIError, anthropic.APIConnectionError) as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BACKOFF_BASE ** attempt)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Judge returned malformed JSON: {text!r}") from exc
        raise RuntimeError(f"Judge API failed after {self.MAX_RETRIES} attempts") from last_exc

    def evaluate(self, question: str, context: str, response: str,
                 expected: str, dimensions: list[str] | None = None) -> JudgeReport:
        from evals.layer2_judgment.prompts.judge_prompts import get_prompt
        dims = dimensions or list(RUBRIC_WEIGHTS)
        scores, weighted_sum = [], 0.0
        for dim in dims:
            weight = RUBRIC_WEIGHTS.get(dim, 0)
            prompt = get_prompt(dim, question=question, context=context,
                                response=response, expected=expected)
            data   = self._call_with_retry(prompt)
            raw    = max(1, min(5, int(data["score"])))
            norm   = (raw - 1) / 4
            scores.append(JudgeScore(dim, raw, round(norm, 3),
                                     JUDGE_THRESHOLD, norm >= JUDGE_THRESHOLD,
                                     data.get("reasoning", ""), self.model))
            weighted_sum += weight * norm
        return JudgeReport(scores, round(weighted_sum, 3),
                           weighted_sum >= JUDGE_THRESHOLD, self.model)


# ── Factory ────────────────────────────────────────────────────────────────────

def get_judge(use_mock: bool | None = None,
              dimensions: list[str] | None = None) -> MockClaudeJudge | ClaudeJudge:
    """
    Return the appropriate judge instance.

    Resolution order:
      1. use_mock argument (explicit override)
      2. USE_MOCK_JUDGE env var  ("true" / "1" / "yes")
      3. Presence of ANTHROPIC_API_KEY  (key set → real; missing → mock + warning)
    """
    if use_mock is None:
        env_val = os.getenv("USE_MOCK_JUDGE", "").lower()
        if env_val in ("1", "true", "yes"):
            use_mock = True
        else:
            has_key  = bool(os.getenv("ANTHROPIC_API_KEY", ""))
            use_mock = not has_key
            if not has_key:
                import warnings
                warnings.warn(
                    "ANTHROPIC_API_KEY not set – falling back to MockClaudeJudge. "
                    "Set the key or USE_MOCK_JUDGE=true to silence this warning.",
                    stacklevel=2,
                )
    return MockClaudeJudge() if use_mock else ClaudeJudge()
