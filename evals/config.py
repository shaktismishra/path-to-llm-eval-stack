"""
Central configuration for the 3-layer eval stack.
All settings read from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
GOLDEN_SET_DIR = REPO_ROOT / "evals" / "layer1_ground_truth" / "golden_set"
FEEDBACK_LOG_PATH = REPO_ROOT / os.getenv(
    "FEEDBACK_LOG_PATH", "evals/layer3_feedback/storage/feedback_log.jsonl"
)
REPORTS_DIR = REPO_ROOT / "reports"

# ── Anthropic ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Judge models ───────────────────────────────────────────────────────────────
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL_FAST: str = os.getenv("JUDGE_MODEL_FAST", "claude-haiku-4-5-20251001")

# ── Passing thresholds (0.0 – 1.0) ────────────────────────────────────────────
THRESHOLDS: dict[str, float] = {
    "faithfulness": float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7")),
    "relevance": float(os.getenv("RELEVANCE_THRESHOLD", "0.7")),
    "completeness": float(os.getenv("COMPLETENESS_THRESHOLD", "0.6")),
    "correctness": float(os.getenv("CORRECTNESS_THRESHOLD", "0.7")),
    # Overall weighted average threshold
    "overall": 0.70,
}

# ── Layer 3: auto-promotion ────────────────────────────────────────────────────
AUTO_PROMOTE_THRESHOLD: float = float(os.getenv("AUTO_PROMOTE_THRESHOLD", "0.85"))

# ── Rubric weights (must sum to 1.0) ──────────────────────────────────────────
RUBRIC_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.35,   # Most important for RAG: don't hallucinate
    "relevance": 0.25,      # Did we actually answer the question?
    "completeness": 0.20,   # Did we cover all key points?
    "correctness": 0.20,    # Is the factual content right?
}
