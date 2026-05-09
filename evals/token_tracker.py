"""
Token usage tracking across evaluation layers.
Approximation: 1 token ~ 4 characters (Anthropic / OpenAI rule of thumb).
Layers 1, 2a, and 3 show text processed; Layer 2b shows real API token cost.
"""
from __future__ import annotations
from dataclasses import dataclass, field


def count_tokens(text: str) -> int:
    """Approximate token count: len(text) // 4, minimum 1."""
    return max(1, len(text) // 4)


@dataclass
class LayerTokens:
    layer: str
    input_tokens: int  = 0
    output_tokens: int = 0
    llm_call: bool     = False   # True only for layers that hit the LLM API

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def note(self) -> str:
        return "API call" if self.llm_call else "no LLM call"


def render_token_table(layers: list[LayerTokens]) -> str:
    """Return a formatted token-usage table as a string."""
    grand_in  = sum(t.input_tokens  for t in layers)
    grand_out = sum(t.output_tokens for t in layers)
    grand_tot = grand_in + grand_out
    lines = [
        f"  {'Token Usage':─<65}",
        f"  {'Layer':<28} {'In':>7}  {'Out':>7}  {'Total':>7}  Type",
        f"  {'-'*28} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*12}",
    ]
    for t in layers:
        flag = " *" if t.llm_call else ""
        lines.append(
            f"  {t.layer:<28} {t.input_tokens:>7,}  {t.output_tokens:>7,}"
            f"  {t.total:>7,}  {t.note()}{flag}"
        )
    lines += [
        f"  {'─'*28} {'─'*7}  {'─'*7}  {'─'*7}",
        f"  {'TOTAL':<28} {grand_in:>7,}  {grand_out:>7,}  {grand_tot:>7,}",
        f"",
        f"  * Layer 2b tokens = real API usage in production",
        f"    (MockClaudeJudge active here – no API call made)",
    ]
    return "\n".join(lines)
