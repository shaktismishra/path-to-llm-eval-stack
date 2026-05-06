"""
Layer 2 — LLM-as-Judge: Prompt templates.

The article's groundedness prompt is included verbatim and extended with
full rubrics for Faithfulness, Relevance, Completeness, and Correctness.

CRITICAL CALIBRATION NOTES (from the article):
  - LLM judges have measurement noise — they can drift when the judge model is updated.
  - They can reward fluent answers that are still factually wrong.
  - Calibrate by starting with 100–200 human-labeled examples and comparing
    judge scores against human scores. Track the noise floor.
  - Lock the judge model version. Monitor when scores move for unrelated reasons.
  - LLM-as-judge is a SCALE TOOL, not a source of truth.
"""
from __future__ import annotations


# ── Core judge prompt (from the article) ──────────────────────────────────────

GROUNDEDNESS_PROMPT = """\
You are evaluating an AI agent's response for groundedness.

Source documents:
{context}

Agent response:
{response}

Score the response on a scale of 1-5 for groundedness:
5 = Every claim is directly supported by the source documents
3 = Most claims supported, minor extrapolations present
1 = Contains claims not present in or contradicted by source documents

Return JSON only: {{"score": int, "reason": str}}"""


# ── Extended rubric prompts ────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """\
You are an expert evaluator assessing whether an AI agent's answer is \
faithful to the provided source documents. Faithful means: the answer \
does not introduce information not present in or contradicted by the context.

Question: {question}

Source context:
{context}

Agent answer:
{response}

Evaluate faithfulness on a 1–5 scale:
  5 = Every claim in the answer is directly supported by the context. No hallucinations.
  4 = Almost all claims are supported; one minor unverified detail.
  3 = Most claims supported, but 1-2 notable extrapolations or unsupported claims.
  2 = Several claims not in context, or one clearly contradicted claim.
  1 = Answer contains significant information not in context or contradicts context.

Your response MUST be valid JSON only. Do not include markdown, code fences, or explanation outside JSON.
Return: {{"score": int, "reason": str, "unsupported_claims": list[str]}}"""


RELEVANCE_PROMPT = """\
You are an expert evaluator assessing whether an AI agent's answer \
actually addresses the user's question.

Question: {question}

Agent answer:
{response}

Evaluate relevance on a 1–5 scale:
  5 = The answer directly and completely addresses what was asked.
  4 = Mostly on-topic; minor tangential content.
  3 = Partially relevant — addresses some aspects but misses key parts of the question.
  2 = Largely off-topic; the user would need to ask again.
  1 = Does not address the question at all.

Your response MUST be valid JSON only.
Return: {{"score": int, "reason": str, "missed_aspects": list[str]}}"""


COMPLETENESS_PROMPT = """\
You are an expert evaluator assessing whether an AI agent's answer \
covers all key points from the provided context that are relevant to the question.

Question: {question}

Source context:
{context}

Expected key points (from ground truth):
{expected_answer}

Agent answer:
{response}

Evaluate completeness on a 1–5 scale:
  5 = All key points from the expected answer are present in the agent's answer.
  4 = Almost all key points present; one minor omission.
  3 = Several key points present; 1-2 important omissions.
  2 = Many key points missing; only partial answer.
  1 = Most key points absent; severely incomplete.

Your response MUST be valid JSON only.
Return: {{"score": int, "reason": str, "missing_key_points": list[str]}}"""


CORRECTNESS_PROMPT = """\
You are an expert evaluator assessing factual correctness of an AI agent's answer, \
comparing it against a known ground truth.

Question: {question}

Ground truth (correct answer):
{expected_answer}

Agent answer:
{response}

Evaluate correctness on a 1–5 scale:
  5 = Factually identical to ground truth; no errors.
  4 = Correct on all major facts; one minor inaccuracy.
  3 = Mostly correct; 1-2 factual errors that could mislead.
  2 = Several factual errors; the answer would mislead the user.
  1 = Fundamentally incorrect; the main facts are wrong.

Your response MUST be valid JSON only.
Return: {{"score": int, "reason": str, "factual_errors": list[str]}}"""


# ── Tool-call accuracy (for agentic patterns) ─────────────────────────────────

TOOL_CALL_ACCURACY_PROMPT = """\
You are an expert evaluator assessing whether an AI agent called the \
correct tool(s) with the correct arguments to fulfil a user request.

User request: {question}

Available tools: {available_tools}

Agent's tool calls (JSON):
{tool_calls}

Expected tool calls (from ground truth):
{expected_tool_calls}

Evaluate tool-call accuracy on a 1–5 scale:
  5 = Called the exact right tool(s) with correct arguments.
  4 = Right tool, mostly correct arguments; minor parameter issue.
  3 = Right tool category but wrong specific tool, or key argument wrong.
  2 = Wrong tool called; task would likely fail.
  1 = No tool called when one was needed, or completely wrong tool.

Your response MUST be valid JSON only.
Return: {{"score": int, "reason": str, "issues": list[str]}}"""


# ── Registry ──────────────────────────────────────────────────────────────────

PROMPT_REGISTRY: dict[str, str] = {
    "groundedness": GROUNDEDNESS_PROMPT,
    "faithfulness": FAITHFULNESS_PROMPT,
    "relevance": RELEVANCE_PROMPT,
    "completeness": COMPLETENESS_PROMPT,
    "correctness": CORRECTNESS_PROMPT,
    "tool_call_accuracy": TOOL_CALL_ACCURACY_PROMPT,
}


def get_prompt(dimension: str) -> str:
    if dimension not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown judge dimension '{dimension}'. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[dimension]
