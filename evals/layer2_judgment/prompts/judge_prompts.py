"""
Scoring rubric prompts for the LLM-as-judge.
Each prompt returns structured JSON: {"score": 1-5, "reasoning": "..."}

Calibration note: lock the judge model version and validate against
100-200 human-labelled examples before relying on these scores in CI.
"""
from __future__ import annotations

_PROMPTS: dict[str, str] = {

    "faithfulness": """You are an expert evaluator assessing whether an AI agent's response
is faithful to the provided source context.

Question: {question}
Context: {context}
Agent Response: {response}

Score the FAITHFULNESS of the response on a 1-5 scale:
1 = Major hallucinations or contradictions with the context
2 = Some unsupported claims introduced
3 = Mostly faithful with minor unsupported details
4 = Faithful; all claims traceable to context
5 = Perfectly faithful; response only uses information from context

Return JSON only: {{"score": <int>, "reasoning": "<one sentence>"}}""",

    "relevance": """You are an expert evaluator assessing whether an AI agent's response
directly addresses the user's question.

Question: {question}
Context: {context}
Agent Response: {response}

Score the RELEVANCE of the response on a 1-5 scale:
1 = Completely off-topic
2 = Tangentially related
3 = Partially addresses the question
4 = Mostly addresses the question with minor gaps
5 = Directly and completely addresses the question

Return JSON only: {{"score": <int>, "reasoning": "<one sentence>"}}""",

    "completeness": """You are an expert evaluator assessing whether an AI agent's response
covers all essential information needed to fully answer the question.

Question: {question}
Context: {context}
Agent Response: {response}
Expected Answer: {expected}

Score the COMPLETENESS on a 1-5 scale:
1 = Missing most essential information
2 = Missing several important points
3 = Covers main points but missing some details
4 = Covers all essential points with minor omissions
5 = Comprehensive coverage of all relevant information

Return JSON only: {{"score": <int>, "reasoning": "<one sentence>"}}""",

    "correctness": """You are an expert evaluator assessing the factual correctness of
an AI agent's response against the expected ground-truth answer.

Question: {question}
Agent Response: {response}
Expected Answer: {expected}

Score the CORRECTNESS on a 1-5 scale:
1 = Contains multiple factual errors
2 = Contains significant factual errors
3 = Mostly correct with some inaccuracies
4 = Correct with very minor inaccuracies
5 = Completely correct, matches expected answer

Return JSON only: {{"score": <int>, "reasoning": "<one sentence>"}}""",
}


def get_prompt(dimension: str, **kwargs) -> str:
    """Format a judge prompt for the given dimension."""
    if dimension not in _PROMPTS:
        raise ValueError(f"Unknown dimension '{dimension}'. Available: {list(_PROMPTS)}")
    return _PROMPTS[dimension].format(**kwargs)
