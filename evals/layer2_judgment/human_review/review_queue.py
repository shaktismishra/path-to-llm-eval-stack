"""
Layer 2 — Judgment Pattern 3: Human-in-the-Loop Review.

Non-negotiable for the highest-risk decisions:
  - Medical recommendations
  - Legal language
  - Financial advice
  - Regulated workflows
  - Customer-impacting policy decisions

The article's sampling strategy — you don't review everything, you sample:
  - A percentage of production traffic weekly
  - High-risk flows and low-confidence outputs
  - New intents the agent hasn't seen before
  - Cases where the LLM judge disagrees with prior patterns

This module manages the human review queue: items are written to a
JSONL file, routed to the right domain owner, and results flow back
into Layer 3 to update the golden set.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from evals.config import REPO_ROOT


REVIEW_QUEUE_PATH = REPO_ROOT / "evals" / "layer2_judgment" / "human_review" / "review_queue.jsonl"


class ReviewPriority(str, Enum):
    CRITICAL = "critical"   # Block deployment immediately
    HIGH = "high"           # Review before next release
    MEDIUM = "medium"       # Review within 1 week
    LOW = "low"             # Review opportunistically


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"    # Agent output was acceptable
    REJECTED = "rejected"    # Agent output was wrong — promote to golden set
    ESCALATED = "escalated"  # Needs legal / compliance / exec input


@dataclass
class ReviewItem:
    """A single item in the human review queue."""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    entry_id: str = ""              # Golden entry ID if from golden set, else ""
    question: str = ""
    context: str = ""
    agent_response: str = ""
    expected_answer: str = ""
    priority: ReviewPriority = ReviewPriority.MEDIUM
    status: ReviewStatus = ReviewStatus.PENDING
    routing_owner: str = ""         # e.g. "compliance-team", "hr-team"
    trigger_reason: str = ""        # Why was this queued? e.g. "llm_judge_low_confidence"
    llm_judge_score: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed_at: str | None = None
    reviewer: str | None = None
    reviewer_notes: str = ""
    promote_to_golden: bool = False  # Set True to feed into Layer 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewItem":
        d = dict(d)
        d["priority"] = ReviewPriority(d.get("priority", "medium"))
        d["status"] = ReviewStatus(d.get("status", "pending"))
        return cls(**d)


class HumanReviewQueue:
    """
    Manages items requiring human judgment.

    The queue is backed by a JSONL file — simple, git-diffable, no database needed.
    In production, swap the storage layer for your team's ticketing system
    (Jira, Linear, etc.) by subclassing and overriding _write / _read.
    """

    def __init__(self, queue_path: Path = REVIEW_QUEUE_PATH) -> None:
        self.queue_path = queue_path
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self.queue_path.touch()

    def enqueue(self, item: ReviewItem) -> str:
        """Add an item to the review queue. Returns the item_id."""
        with self.queue_path.open("a") as f:
            f.write(json.dumps(item.to_dict()) + "\n")
        return item.item_id

    def list_pending(self) -> list[ReviewItem]:
        """Return all items with PENDING status, sorted by priority."""
        priority_order = {
            ReviewPriority.CRITICAL: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3,
        }
        items = [
            item for item in self._read_all()
            if item.status == ReviewStatus.PENDING
        ]
        return sorted(items, key=lambda i: priority_order[i.priority])

    def update_status(self, item_id: str, status: ReviewStatus, **kwargs) -> bool:
        """Update an item's status and optional fields."""
        items = self._read_all()
        updated = False
        for item in items:
            if item.item_id == item_id:
                item.status = status
                item.reviewed_at = datetime.now(timezone.utc).isoformat()
                for k, v in kwargs.items():
                    if hasattr(item, k):
                        setattr(item, k, v)
                updated = True
        if updated:
            self._write_all(items)
        return updated

    def items_to_promote(self) -> list[ReviewItem]:
        """Return rejected items flagged for golden set promotion (Layer 3 handoff)."""
        return [
            item for item in self._read_all()
            if item.status == ReviewStatus.REJECTED and item.promote_to_golden
        ]

    def _read_all(self) -> list[ReviewItem]:
        items = []
        with self.queue_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(ReviewItem.from_dict(json.loads(line)))
        return items

    def _write_all(self, items: list[ReviewItem]) -> None:
        with self.queue_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item.to_dict()) + "\n")

    def stats(self) -> dict:
        items = self._read_all()
        status_counts: dict[str, int] = {}
        for item in items:
            status_counts[item.status.value] = status_counts.get(item.status.value, 0) + 1
        return {
            "total": len(items),
            "by_status": status_counts,
            "to_promote": len(self.items_to_promote()),
        }


# ── Sampling strategy (from the article) ──────────────────────────────────────

def should_queue_for_human_review(
    entry_id: str,
    llm_judge_score: float | None,
    is_high_risk_flow: bool,
    is_new_intent: bool,
    user_escalated: bool,
    low_confidence_threshold: float = 0.5,
) -> tuple[bool, ReviewPriority, str]:
    """
    Decide whether a production trace should be queued for human review,
    and at what priority. Returns (should_queue, priority, reason).

    Sampling weights (from the article):
      - Low-confidence outputs            → HIGH
      - Cases LLM judge flagged uncertain → HIGH
      - User escalations / neg feedback   → CRITICAL
      - New intents                        → MEDIUM
      - High-risk workflows               → HIGH
      - Policy-sensitive responses        → HIGH
    """
    if user_escalated:
        return True, ReviewPriority.CRITICAL, "user_escalation"

    if llm_judge_score is not None and llm_judge_score < low_confidence_threshold:
        return True, ReviewPriority.HIGH, f"llm_judge_low_confidence ({llm_judge_score:.2f})"

    if is_high_risk_flow:
        return True, ReviewPriority.HIGH, "high_risk_flow"

    if is_new_intent:
        return True, ReviewPriority.MEDIUM, "new_intent"

    return False, ReviewPriority.LOW, ""
