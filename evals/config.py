"""
Central configuration for the 3-layer evaluation stack.
Values are resolved in priority order:
  1. Environment variables (highest – set in .env or CI secrets)
  2. eval_config.yaml  (project-level runtime config)
  3. Hard-coded defaults (lowest)
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
EVAL_CONFIG     = ROOT / "eval_config.yaml"
GOLDEN_SET      = ROOT / "evals" / "layer1_ground_truth" / "golden_set"
FEEDBACK_LOG    = ROOT / "evals" / "layer3_feedback" / "storage" / "feedback_log.jsonl"
REPORTS_DIR     = ROOT / "reports"

# ── Anthropic API ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Judge models ───────────────────────────────────────────────────────────────
JUDGE_MODEL      = os.getenv("JUDGE_MODEL",      "claude-sonnet-4-6")
JUDGE_MODEL_FAST = os.getenv("JUDGE_MODEL_FAST", "claude-haiku-4-5-20251001")

# ── Score thresholds ───────────────────────────────────────────────────────────
JUDGE_THRESHOLD        = float(os.getenv("JUDGE_THRESHOLD",        "0.70"))
AUTO_PROMOTE_THRESHOLD = float(os.getenv("AUTO_PROMOTE_THRESHOLD", "0.85"))

# ── Rubric weights (must sum to 1.0) ──────────────────────────────────────────
RUBRIC_WEIGHTS: dict[str, float] = {
    "faithfulness":  0.35,
    "relevance":     0.25,
    "completeness":  0.20,
    "correctness":   0.20,
}

# ── Layer 1 similarity thresholds ─────────────────────────────────────────────
TOKEN_F1_THRESHOLD = 0.40
ROUGE_L_THRESHOLD  = 0.35


def load_eval_config(path: Path = EVAL_CONFIG) -> dict:
    """
    Load eval_config.yaml.  Returns an empty dict if the file is missing
    so callers can always treat the result as a plain dict.
    """
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
