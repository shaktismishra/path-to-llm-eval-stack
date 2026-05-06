"""
Layer 2 — Judgment Pattern 2: LLM-as-Judge (Claude).

Handles qualitative scoring questions that rules cannot answer:
  - Did the response stay grounded in the retrieved data?
  - Was the explanation relevant to the user's actual question?
  - Did the agent ask the right clarifying question?
  - Did the agent call the right tool?

CALIBRATION DISCIPLINE (article requirements):
  1. Start with 100–200 human-labeled examples; compare against judge scores.
  2. Lock the judge model version — record it in every JudgeScore.
  3. Track score drift over time; monitor when scores move for reasons
     unrelated to agent changes (judge model updates, prompt drift).
  4. LLM-as-judge is a SCALE TOOL, not a source of truth.

The judge uses structured JSON output to minimise parsing errors.
Retry with exponential back-off on transient API failures.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from evals.config import ANTHROPIC_API_KEY, JUDGE_MODEL, THRESHOLDS, RUBRIC_WEIGHTS
from evals.layer1_ground_truth.dataset import GoldenEntry
from evals.layer2_judgment.prompts.judge_prompts import get_prompt


@dataclass
class JudgeScore:
    """
    Output of a single LLM-as-judge evaluation.
    The judge_model field is stored deliberately — lock the version and
    alert when it changes, because judge drift can look like agent regression.
    """
    dimension: str      # faithfulness / relevance / completeness / correctness
    raw_score: int      # 1–5 as returned by the judge
    normalised: float   # 0.0–1.0  (raw_score - 1) / 4
    passed: bool        # normalised >= threshold
    threshold: float
    reason: str
    details: dict = field(default_factory=dict)  # extra keys from judge response
    judge_model: str = JUDGE_MODEL
    latency_ms: int = 0

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} [{self.dimension}] "
            f"score={self.raw_score}/5 ({self.normalised:.2f}) "
            f"threshold={self.threshold} | {self.reason}"
        )


@dataclass
class JudgeReport:
    """Aggregated result across all judge dimensions for one golden entry."""
    entry_id: str
    scores: list[JudgeScore]
    weighted_average: float
    overall_passed: bool
    judge_model: str = JUDGE_MODEL

    @property
    def failed_dimensions(self) -> list[str]:
        return [s.dimension for s in self.scores if not s.passed]

    def summary(self) -> str:
        lines = [f"Entry: {self.entry_id} | Overall: {'PASS ✅' if self.overall_passed else 'FAIL ❌'}"]
        lines.append(f"Weighted average: {self.weighted_average:.3f}")
        for s in self.scores:
            lines.append(f"  {s}")
        return "\n".join(lines)


class ClaudeJudge:
    """
    LLM-as-judge backed by Anthropic Claude.

    Usage:
        judge = ClaudeJudge()
        report = judge.evaluate(entry=entry, response="The API rate limit is 60 RPM...")
        print(report.summary())
    """

    def __init__(self, model: str = JUDGE_MODEL, api_key: str = ANTHROPIC_API_KEY) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Check your .env file.")
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_judge(self, prompt: str) -> dict:
        """Single judge API call with retry on transient failures."""
        t0 = time.monotonic()
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        raw = message.content[0].text.strip()

        # Strip markdown code fences if present (some models add them despite the prompt)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Judge returned invalid JSON: {e}\nRaw output: {raw}") from e

        result["_latency_ms"] = latency_ms
        return result

    def score_dimension(
        self,
        dimension: str,
        question: str,
        context: str,
        response: str,
        expected_answer: str = "",
    ) -> JudgeScore:
        """
        Score a single dimension (faithfulness / relevance / completeness / correctness).
        The rubric prompts are in layer2_judgment/prompts/judge_prompts.py.
        """
        prompt_template = get_prompt(dimension)
        prompt = prompt_template.format(
            question=question,
            context=context,
            response=response,
            expected_answer=expected_answer,
        )

        result = self._call_judge(prompt)
        raw_score = int(result.get("score", 1))
        # Clamp to 1–5
        raw_score = max(1, min(5, raw_score))
        normalised = (raw_score - 1) / 4.0
        threshold = THRESHOLDS.get(dimension, 0.7)

        return JudgeScore(
            dimension=dimension,
            raw_score=raw_score,
            normalised=normalised,
            passed=normalised >= threshold,
            threshold=threshold,
            reason=result.get("reason", ""),
            details={k: v for k, v in result.items() if k not in ("score", "reason", "_latency_ms")},
            judge_model=self.model,
            latency_ms=result.get("_latency_ms", 0),
        )

    def evaluate(self, entry: GoldenEntry, response: str) -> JudgeReport:
        """
        Run all LLM judge dimensions specified in the golden entry's judgment_patterns.
        Dimensions not listed in the entry's judgment_patterns are skipped.
        """
        dimensions_to_run = entry.llm_judge_dimensions()
        if not dimensions_to_run:
            # Default: run faithfulness and relevance for every RAG entry
            dimensions_to_run = ["faithfulness", "relevance"]

        scores: list[JudgeScore] = []
        for dimension in dimensions_to_run:
            score = self.score_dimension(
                dimension=dimension,
                question=entry.question,
                context=entry.context,
                response=response,
                expected_answer=entry.expected_answer,
            )
            scores.append(score)

        # Weighted average using configured weights (defaults to equal weight if not in registry)
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scores:
            w = RUBRIC_WEIGHTS.get(s.dimension, 1.0 / len(scores))
            weighted_sum += s.normalised * w
            total_weight += w
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

        return JudgeReport(
            entry_id=entry.id,
            scores=scores,
            weighted_average=weighted_avg,
            overall_passed=weighted_avg >= THRESHOLDS["overall"],
            judge_model=self.model,
        )
