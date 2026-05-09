"""
Context retrieval for the eval stack.

Two modes are supported, set via eval_config.yaml retrieval.mode:

  keyword    (default) – TF-IDF-style word-overlap scoring over the Markdown FAQ.
             Requires docs/faq.md (or the path in retrieval.faq_path).

  structured – Loads a JSON or YAML file with explicit Q/A pairs, enabling
               exact-match lookups and richer scoring.
             Requires docs/faq.json (or the path in retrieval.structured_path).
             Falls back to keyword mode if the structured file is not found.

Replace retrieve_context() with a vector-store call for production-grade retrieval.
See README.md § "Wiring stronger retrieval" for a ChromaDB example.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_faq_sections(faq_path: Path) -> list[tuple[str, str]]:
    """Parse a Markdown FAQ file into (heading, body) pairs."""
    text = faq_path.read_text(encoding="utf-8")
    sections: list[tuple[str, str]] = []
    current_heading, current_body = "", []
    for line in text.splitlines():
        if line.startswith("**Q:"):
            if current_heading:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip("*").strip()
            current_body    = []
        elif current_heading:
            current_body.append(line)
    if current_heading:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return sections


def load_structured_faq(path: Path) -> list[tuple[str, str]]:
    """
    Load a structured FAQ from a JSON or YAML file.

    Expected JSON format (list of objects):
        [{"question": "...", "answer": "..."}, ...]

    or wrapped in a top-level dict:
        {"title": "...", "entries": [{"question": "...", "answer": "..."}, ...]}

    YAML format uses the same schema.
    """
    raw = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional dep; only needed for structured YAML FAQs
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    entries = data if isinstance(data, list) else data.get("entries", [])
    return [(item["question"], item["answer"]) for item in entries]


# ── Scoring ────────────────────────────────────────────────────────────────────

def _overlap_score(question: str, heading: str, body: str) -> int:
    """Word-overlap between the question and the heading + body."""
    q_words = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
    h_words = set(re.sub(r"[^a-z0-9 ]", " ", heading.lower()).split())
    b_words = set(re.sub(r"[^a-z0-9 ]", " ", body.lower()).split())
    return len(q_words & (h_words | b_words))


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve_context(
    question:        str,
    faq_path:        Path,
    top_k:           int            = 2,
    mode:            str            = "keyword",
    structured_path: Optional[Path] = None,
) -> str:
    """
    Return the top-k most relevant FAQ answers joined as a context string.

    Args:
        question:        The user's question.
        faq_path:        Path to the Markdown FAQ (always used in keyword mode;
                         also the fallback when structured file is absent).
        top_k:           Maximum number of sections to include.
        mode:            "keyword" or "structured".
        structured_path: Path to the structured JSON/YAML FAQ used in structured mode.
                         If None and mode=="structured", inferred as faq_path.with_suffix(".json").

    Returns:
        Joined context string, or "" if nothing relevant is found.
    """
    if mode == "structured":
        # resolve structured path
        s_path = structured_path
        if s_path is None:
            s_path = faq_path.with_suffix(".json")
            if not s_path.exists():
                s_path = faq_path.with_suffix(".yaml")

        if s_path is not None and s_path.exists():
            sections = load_structured_faq(s_path)
        else:
            # graceful fallback to keyword mode
            sections = load_faq_sections(faq_path) if faq_path.exists() else []
    else:
        sections = load_faq_sections(faq_path) if faq_path.exists() else []

    if not sections:
        return ""

    scored: list[tuple[int, str, str]] = [
        (_overlap_score(question, heading, body), heading, body)
        for heading, body in sections
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    chunks = [f"{h}\n{b}" for score, h, b in scored[:top_k] if score > 0]
    return "\n\n".join(chunks) if chunks else ""
