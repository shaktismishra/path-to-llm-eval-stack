"""
Golden dataset loader.
Each entry is a versioned, reviewed YAML file in golden_set/.
Categories: regulated | historical-failures | adversarial
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evals.config import GOLDEN_SET


@dataclass
class GoldenEntry:
    """Single test case from the governed golden set."""
    id: str
    category: str                  # regulated | historical-failures | adversarial
    question: str
    context: str
    expected_answer: str
    must_contain: list   = field(default_factory=list)
    must_not_contain: list = field(default_factory=list)
    judgment_patterns: list = field(default_factory=list)
    owner: str           = "unknown"
    difficulty: str      = "medium"
    # historical-failures extras
    root_cause: Optional[str] = None
    # adversarial extras
    attack_type: Optional[str] = None

    def needs_human_review(self) -> bool:
        return "human_review" in self.judgment_patterns

    def eval_specs(self) -> dict:
        return {
            "must_contain":     self.must_contain,
            "must_not_contain": self.must_not_contain,
            "judgment_patterns": self.judgment_patterns,
        }


class GoldenDataset:
    """Loads and filters all YAML entries from golden_set/."""

    def __init__(self, golden_set_path: Path = GOLDEN_SET):
        self._path   = golden_set_path
        self.entries: list[GoldenEntry] = []

    def load(self) -> "GoldenDataset":
        self.entries = []
        for yaml_file in sorted(self._path.rglob("*.yaml")):
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                self.entries.append(GoldenEntry(**{
                    k: v for k, v in data.items()
                    if k in GoldenEntry.__dataclass_fields__
                }))
        return self

    def filter(self, *, category=None, difficulty=None, owner=None,
               needs_human_review=None) -> list[GoldenEntry]:
        result = self.entries
        if category:
            result = [e for e in result if e.category == category]
        if difficulty:
            result = [e for e in result if e.difficulty == difficulty]
        if owner:
            result = [e for e in result if e.owner == owner]
        if needs_human_review is not None:
            result = [e for e in result if e.needs_human_review() == needs_human_review]
        return result

    def stats(self) -> dict:
        from collections import Counter
        return {
            "total":      len(self.entries),
            "by_category": dict(Counter(e.category   for e in self.entries)),
            "by_difficulty": dict(Counter(e.difficulty for e in self.entries)),
        }

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)
