"""
Layer 1 — Ground Truth: Dataset loader and manager.

The golden set is a GOVERNED ARTIFACT, not a spreadsheet.
  - Version-controlled in git (treat like code)
  - Organised into three categories: regulated / historical-failures / adversarial
  - Each entry is a YAML file owned by a named team
  - Changes go through pull-request review

Usage:
    dataset = GoldenDataset.load()
    regulated = dataset.filter(category="regulated")
    for entry in regulated:
        print(entry.id, entry.question)
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from evals.config import GOLDEN_SET_DIR

GoldenCategory = Literal["regulated", "historical-failures", "adversarial"]
Difficulty = Literal["easy", "medium", "hard"]
JudgmentPattern = Literal["code_evaluator", "llm_judge", "human_review"]


@dataclass
class GoldenEntry:
    """A single entry in the golden dataset."""

    id: str
    category: GoldenCategory
    question: str
    context: str
    expected_answer: str
    difficulty: Difficulty = "medium"
    source: str = "human"
    owner: str = ""
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    judgment_patterns: list[dict] = field(default_factory=list)
    notes: str = ""
    # Historical-failure-specific
    failure_root_cause: str | None = None
    failure_description: str | None = None
    incident_cost: str | None = None
    # Adversarial-specific
    attack_type: str | None = None
    # Metadata
    created_at: str = ""
    version: int = 1
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "GoldenEntry":
        with path.open() as f:
            data = yaml.safe_load(f)
        # judgment_patterns may be a list of dicts with mixed keys
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def requires_human_review(self) -> bool:
        return any(
            p.get("human_review") is True for p in self.judgment_patterns
        )

    def code_evaluators(self) -> list[str]:
        return [
            p["code_evaluator"]
            for p in self.judgment_patterns
            if "code_evaluator" in p
        ]

    def llm_judge_dimensions(self) -> list[str]:
        return [
            p["llm_judge"]
            for p in self.judgment_patterns
            if "llm_judge" in p
        ]


class GoldenDataset:
    """
    The production golden dataset — the ground truth contract for the agent.

    Every eval program needs a written, governed set of cases the agent must
    never get wrong. Build it from three sources:
      1. Regulated edge cases (compliance / legal requirements)
      2. Historical failure cases (incidents converted to regression tests)
      3. Adversarial cases (prompt injection, jailbreak, policy override)
    """

    def __init__(self, entries: list[GoldenEntry]) -> None:
        self._entries = entries

    @classmethod
    def load(
        cls,
        golden_set_dir: Path = GOLDEN_SET_DIR,
        categories: list[GoldenCategory] | None = None,
    ) -> "GoldenDataset":
        """Load all golden entries from the governed YAML files."""
        entries: list[GoldenEntry] = []
        search_dirs = categories or ["regulated", "historical-failures", "adversarial"]

        for category in search_dirs:
            pattern = str(golden_set_dir / category / "*.yaml")
            for yaml_path in sorted(glob.glob(pattern)):
                try:
                    entry = GoldenEntry.from_yaml(Path(yaml_path))
                    entries.append(entry)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load golden entry from {yaml_path}: {e}"
                    ) from e

        return cls(entries)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def filter(
        self,
        category: GoldenCategory | None = None,
        difficulty: Difficulty | None = None,
        owner: str | None = None,
        requires_human_review: bool | None = None,
    ) -> "GoldenDataset":
        """Return a filtered subset of the dataset."""
        entries = self._entries
        if category:
            entries = [e for e in entries if e.category == category]
        if difficulty:
            entries = [e for e in entries if e.difficulty == difficulty]
        if owner:
            entries = [e for e in entries if e.owner == owner]
        if requires_human_review is not None:
            entries = [e for e in entries if e.requires_human_review() == requires_human_review]
        return GoldenDataset(entries)

    def stats(self) -> dict:
        """Summary statistics for monitoring golden set health."""
        categories: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        for e in self._entries:
            categories[e.category] = categories.get(e.category, 0) + 1
            difficulties[e.difficulty] = difficulties.get(e.difficulty, 0) + 1
        return {
            "total": len(self._entries),
            "by_category": categories,
            "by_difficulty": difficulties,
            "require_human_review": sum(1 for e in self._entries if e.requires_human_review()),
        }
