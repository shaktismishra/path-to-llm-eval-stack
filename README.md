# LLM Eval Stack

A **3-layer evaluation framework** for RAG-based FAQ agents — deterministic gates, LLM-as-judge scoring, and a production feedback loop that converts live failures into regression tests.

Based on the architecture described in [path-to-llm-eval-stack](https://github.com/shaktismishra/path-to-llm-eval-stack).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running Evaluations](#running-evaluations)
6. [Production Checklist](#production-checklist)
7. [Adding Golden Entries](#adding-golden-entries)
8. [Switching to the Real LLM Judge](#switching-to-the-real-llm-judge)
9. [Wiring Your Own Agent](#wiring-your-own-agent)
10. [Swapping to Vector Retrieval](#swapping-to-vector-retrieval)
11. [Persisting and Triaging Feedback](#persisting-and-triaging-feedback)
12. [Running Tests](#running-tests)
13. [Token Usage](#token-usage)
14. [Environment Variables Reference](#environment-variables-reference)

---

## Architecture

```
Question
   │
   ▼
[Retrieval]  docs/faq.md  →  relevant context chunks
   │
   ▼
[Agent]  faq_agent(question, context)  →  response
   │
   ├──▶  Layer 1: Ground Truth
   │         must_contain / must_not_contain  (hard gates)
   │         token_f1 / rouge_l              (similarity)
   │         evals/layer1_ground_truth/
   │
   ├──▶  Layer 2: Judgment
   │         2a. Code Evaluators  (deterministic regex rules)
   │         2b. LLM-as-Judge     (Claude – weighted rubric scoring)
   │         evals/layer2_judgment/
   │
   └──▶  Layer 3: Feedback Loop
             ProductionFeedbackCollector  →  JSONL log
             FailurePatternAnalyzer       →  9 root-cause clusters
             GoldenSetUpdater             →  YAML regression tests
             evals/layer3_feedback/
```

### The three questions this stack answers

| # | Question | Mechanism |
|---|---|---|
| 1 | Does the response contain what it must? | Layer 1 constraint gates |
| 2 | Is the response qualitatively good? | Layer 2 code evaluators + LLM judge |
| 3 | Are we catching regressions from production? | Layer 3 feedback loop |

---

## Project Structure

```
.
├── demo_eval_agent.py                    # Quick-start demo (uses EvalRunner)
├── run_evals.py                          # Production CLI entry point
├── eval_config.yaml                      # Runtime configuration (edit this)
├── pyproject.toml                        # Package definition + dependencies
├── .env.example                          # Environment variable template
│
├── docs/
│   └── faq.md                            # FAQ knowledge base (source of truth)
│
├── evals/
│   ├── config.py                         # Config loader (env vars + YAML)
│   ├── retrieval.py                      # Keyword retrieval over docs/faq.md
│   ├── runner.py                         # EvalRunner orchestrator
│   ├── token_tracker.py                  # Token counting + table rendering
│   │
│   ├── layer1_ground_truth/
│   │   ├── dataset.py                    # GoldenEntry, GoldenDataset
│   │   ├── metrics.py                    # must_contain, token_f1, rouge_l
│   │   └── golden_set/
│   │       ├── regulated/                # Compliance test cases
│   │       ├── historical-failures/      # Past incidents as regression tests
│   │       └── adversarial/              # Prompt injection / jailbreak tests
│   │
│   ├── layer2_judgment/
│   │   ├── code_evaluators/evaluators.py # Regex rule checks + EVALUATOR_REGISTRY
│   │   ├── llm_judge/judge.py            # MockClaudeJudge, ClaudeJudge, get_judge()
│   │   └── prompts/judge_prompts.py      # Rubric prompts (faithfulness/relevance/…)
│   │
│   └── layer3_feedback/
│       ├── collector.py                  # ProductionFeedbackCollector → JSONL
│       ├── analyzer.py                   # FailurePatternAnalyzer (9 clusters)
│       ├── updater.py                    # GoldenSetUpdater → YAML promotion
│       └── storage/
│           └── feedback_log.jsonl        # Persisted production traces (gitignored)
│
└── tests/
    ├── conftest.py                       # Shared fixtures
    ├── test_layer1.py                    # 24 tests for metrics
    ├── test_layer2.py                    # 32 tests for evaluators + judge
    └── test_layer3.py                    # 30 tests for collector/analyzer/updater
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd llm-eval-stack
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY if you have one

# 3. Run the demo
python -X utf8 demo_eval_agent.py

# 4. Run the full CLI
python run_evals.py run

# 5. Run the test suite
pytest tests/ -v
```

---

## Configuration

All runtime behaviour is controlled by two files.

### `eval_config.yaml`

```yaml
agent:
  name: faq-bot
  domain: FAQ Knowledge Base Bot

retrieval:
  faq_path: docs/faq.md   # path to your FAQ document (relative to project root)
  top_k: 2                # how many FAQ sections to surface per question

golden_set:
  path: evals/layer1_ground_truth/golden_set
  categories:
    - regulated
    - historical-failures
    - adversarial

evaluators:
  code:                                     # list of EVALUATOR_REGISTRY keys to run
    - eval_ssn_redacted
    - eval_no_policy_override
    - eval_no_optimistic_delivery_date
    - eval_no_unauthorized_return_window
  judge:
    # use_mock: true                        # uncomment to force mock regardless of API key
    dimensions:
      - faithfulness
      - relevance
      - completeness
      - correctness

feedback:
  persist: true
  log_path: evals/layer3_feedback/storage/feedback_log.jsonl
  auto_promote_threshold: 0.85
```

### `.env`

```bash
ANTHROPIC_API_KEY=sk-ant-...      # required for the real LLM judge
USE_MOCK_JUDGE=false              # set true to force MockClaudeJudge everywhere
JUDGE_MODEL=claude-sonnet-4-6    # override the judge model
JUDGE_THRESHOLD=0.70              # minimum weighted score to pass Layer 2b
AUTO_PROMOTE_THRESHOLD=0.85       # traces below this are auto-flagged for triage
```

---

## Running Evaluations

### Demo (quick visual walkthrough)

```bash
python -X utf8 demo_eval_agent.py
```

Shows all three layers with per-metric detail and a token usage table. Does not persist traces to disk.

### CLI (`run_evals.py`)

```bash
# Run all golden entries
python run_evals.py run

# Run a single entry by ID
python run_evals.py run --id faq-reg-001

# Run only regulated entries
python run_evals.py run --category regulated

# One-line summary per entry (no per-layer detail)
python run_evals.py run --summary

# Show feedback log statistics
python run_evals.py status

# Cluster recent failure candidates
python run_evals.py triage

# Promote a specific failure to the golden set
python run_evals.py promote --id trace-faq-reg-001 --answer "Correct answer text here."
```

### Programmatic API

```python
from evals.runner import EvalRunner
from evals.layer1_ground_truth.dataset import GoldenEntry

runner = EvalRunner()                          # loads eval_config.yaml
result = runner.run(entry, agent_fn=my_agent)

print(result.all_passed)                       # True / False
print(result.judge_report.weighted_average)    # e.g. 0.812
print(result.failure_cluster)                  # None or "poor_reasoning" etc.
print(EvalRunner.render_token_table(result.token_layers))

# Triage failures logged so far
clusters = runner.triage()

# Promote flagged failures (supply human-validated answers)
promoted = runner.promote_failures({
    "trace-faq-reg-001": "Correct answer here."
})
```

---

## Production Checklist

Work through these items in order to take the stack from demo to production.

### 1. Set your API key

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
USE_MOCK_JUDGE=false
```

`get_judge()` auto-detects: if `ANTHROPIC_API_KEY` is set it returns `ClaudeJudge`; if not it falls back to `MockClaudeJudge` with a warning.

### 2. Wire your real agent

Open `run_evals.py` and replace the placeholder `faq_agent` with your real callable:

```python
# run_evals.py  (line ~35)
# BEFORE
def faq_agent(question: str, context: str) -> str:
    return "... canned response ..."

# AFTER
from my_package.agent import generate_answer   # your real agent
faq_agent = generate_answer
```

The agent signature must be `(question: str, context: str) -> str`.
`EvalRunner.run()` calls it after retrieval.

### 3. Swap keyword retrieval for a vector store

`evals/retrieval.py` exports `retrieve_context(question, faq_path, top_k)`.
Replace the body with a call to your vector store:

```python
# evals/retrieval.py
def retrieve_context(question: str, faq_path: Path, top_k: int = 2) -> str:
    # Example: ChromaDB
    import chromadb
    client     = chromadb.Client()
    collection = client.get_collection("faq")
    results    = collection.query(query_texts=[question], n_results=top_k)
    return "\n\n".join(results["documents"][0])
```

No other file needs to change — `EvalRunner` calls `retrieve_context` directly.

### 4. Expand the golden set

Add YAML files for your domain under `evals/layer1_ground_truth/golden_set/`.
See [Adding Golden Entries](#adding-golden-entries) for the schema.

Regulated entries (compliance, legal, PII) should hard-fail CI.
Historical-failure entries document past incidents.
Adversarial entries cover prompt injection and jailbreak attempts.

### 5. Add domain-specific code evaluators

Add functions to `evals/layer2_judgment/code_evaluators/evaluators.py` and register them:

```python
def eval_no_overdraft_fee_waived(response: str) -> EvalResult:
    """Block agent from promising to waive overdraft fees."""
    if re.search(r"waive.*overdraft|overdraft.*waive", response, re.IGNORECASE):
        return EvalResult(False, "Unauthorised overdraft waiver promised",
                          evaluator="eval_no_overdraft_fee_waived")
    return EvalResult(True, "No unauthorised fee waivers", evaluator="eval_no_overdraft_fee_waived")

EVALUATOR_REGISTRY["eval_no_overdraft_fee_waived"] = eval_no_overdraft_fee_waived
```

Then add the key to `eval_config.yaml` under `evaluators.code`.

### 6. Calibrate the LLM judge

The default rubric weights are:

| Dimension | Weight | Rationale |
|---|---|---|
| Faithfulness | 0.35 | Prevents RAG hallucination — highest priority |
| Relevance | 0.25 | Response must address the actual question |
| Completeness | 0.20 | All essential points covered |
| Correctness | 0.20 | Matches ground-truth facts |

To recalibrate:

1. Collect 100–200 human-labelled (question, response, score) examples.
2. Run `MockClaudeJudge` — or `ClaudeJudge` — on the same set.
3. Adjust `RUBRIC_WEIGHTS` in `evals/config.py` to minimise deviation from human scores.
4. Rerun `pytest tests/test_layer2.py` — the weighted-average maths test will catch drift.

Lock `JUDGE_MODEL` in `.env` once calibrated. A model update resets calibration.

### 7. Set a threshold that matches your risk tolerance

`JUDGE_THRESHOLD=0.70` is a safe starting point for general FAQ use.
Lower it (e.g. `0.65`) for internal tooling where some hallucination is acceptable.
Raise it (e.g. `0.80`) for regulated domains (finance, healthcare, legal).

### 8. Wire the feedback loop to your production system

In production your inference layer should call the collector directly:

```python
from evals.layer3_feedback.collector import (
    ProductionFeedbackCollector, ProductionTrace, FeedbackSignal
)

collector = ProductionFeedbackCollector(
    persist=True,
    log_path=Path("/var/log/evals/feedback_log.jsonl"),
)

# After each inference:
collector.log(ProductionTrace(
    trace_id        = request_id,
    question        = user_question,
    context         = retrieved_context,
    agent_response  = agent_output,
    feedback_signal = FeedbackSignal.POSITIVE if resolved else FeedbackSignal.NEGATIVE,
    llm_judge_score = judge_score,    # run the judge async if latency is a concern
))
```

Run `python run_evals.py triage` weekly to surface clustered failures.

### 9. Set up CI

Add a GitHub Actions step (or equivalent) that runs on every PR touching `evals/`:

```yaml
# .github/workflows/eval-ci.yml
- name: Run eval suite
  run: |
    pip install -e ".[dev]"
    pytest tests/ -m "not integration" -v
    python run_evals.py run --summary
  env:
    USE_MOCK_JUDGE: "true"   # no API cost in CI; use real judge in nightly runs
```

For a nightly real-judge run, set `ANTHROPIC_API_KEY` as a CI secret and drop `USE_MOCK_JUDGE`.

### 10. Promote failures weekly

After triage, validate the correct answers for flagged traces and promote them:

```bash
python run_evals.py triage                    # see what failed and why

python run_evals.py promote \
  --id trace-faq-reg-001 \
  --answer "Correct answer text for this question."
```

The promoted YAML entry is printed to stdout. Paste it into the appropriate
`golden_set/` subdirectory, open a PR, and the next CI run will include it as a
regression test.

---

## Adding Golden Entries

Create a `.yaml` file in the appropriate subdirectory of
`evals/layer1_ground_truth/golden_set/`.

**Schema:**

```yaml
id: faq-reg-002                           # unique — prefix: reg-, hf-, adv-
category: regulated                        # regulated | historical-failures | adversarial
owner: faq-team                            # team responsible for this case
difficulty: medium                         # easy | medium | hard
question: "Can I return a damaged item marked as final sale?"
context: >
  Final-sale items are non-returnable under standard policy. However, items
  that arrive damaged or defective are eligible for a full refund or exchange
  regardless of final-sale designation.
expected_answer: >
  Yes. Even though this is a final-sale item, damaged or defective goods are
  always eligible for a full refund or exchange. Contact support within 48 hours
  with photos of the damage.
must_contain:
  - "damaged"
  - "refund or exchange"
  - "48 hours"
must_not_contain:
  - "no returns"
  - "all sales final"
judgment_patterns:
  - code_evaluator
  - llm_judge
# For historical-failures entries, add:
# root_cause: policy_exception_missed
# For adversarial entries, add:
# attack_type: prompt_injection
```

**Category guide:**

| Category | Use for | CI behaviour |
|---|---|---|
| `regulated` | Compliance, legal, PII, TILA | Hard-fail — blocks merge |
| `historical-failures` | Past production incidents | Regression — blocks merge |
| `adversarial` | Prompt injection, jailbreaks | Hard-fail — blocks merge |

**Judgment pattern guide:**

| Pattern | When to use |
|---|---|
| `code_evaluator` | A regex rule can answer it deterministically |
| `llm_judge` | Qualitative assessment needed |
| `human_review` | Regulated decision — requires human sign-off before merge |

---

## Switching to the Real LLM Judge

| Scenario | What to do |
|---|---|
| No API key (default) | `MockClaudeJudge` is used automatically — no action needed |
| API key available | Set `ANTHROPIC_API_KEY` in `.env` — `get_judge()` returns `ClaudeJudge` |
| Force mock even with key | Set `USE_MOCK_JUDGE=true` in `.env` |
| Force mock for one run | Add `use_mock: true` under `evaluators.judge` in `eval_config.yaml` |
| Force mock in code | `get_judge(use_mock=True)` |

`ClaudeJudge` calls the Anthropic Messages API with up to 3 retries and exponential back-off. It is implemented in `evals/layer2_judgment/llm_judge/judge.py`.

To change the judge model, set `JUDGE_MODEL=claude-opus-4-7` in `.env`.

---

## Wiring Your Own Agent

`EvalRunner.run()` accepts any callable with the signature:

```python
def my_agent(question: str, context: str) -> str:
    ...
```

Pass it directly:

```python
from evals.runner import EvalRunner
from my_package import my_agent

runner = EvalRunner()
result = runner.run(entry, agent_fn=my_agent)
```

Or replace the placeholder in `run_evals.py`:

```python
# run_evals.py
from my_package import my_agent as faq_agent   # same signature required
```

The runner calls `retrieve_context` first, then passes `(question, context)` to the agent. If retrieval returns nothing, it falls back to `entry.context` from the golden YAML.

---

## Swapping to Vector Retrieval

The current implementation in `evals/retrieval.py` uses keyword-overlap scoring — fast and dependency-free, but not semantically aware. Replace `retrieve_context` with a vector search for production:

**ChromaDB example:**

```python
# evals/retrieval.py
import chromadb
from pathlib import Path

_client     = chromadb.PersistentClient(path=".chroma")
_collection = _client.get_or_create_collection("faq")

def retrieve_context(question: str, faq_path: Path, top_k: int = 2) -> str:
    results = _collection.query(query_texts=[question], n_results=top_k)
    return "\n\n".join(results["documents"][0])
```

**Index build script (run once):**

```python
from evals.retrieval import load_faq_sections
from pathlib import Path
import chromadb

sections   = load_faq_sections(Path("docs/faq.md"))
client     = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("faq")
collection.add(
    ids       = [str(i) for i in range(len(sections))],
    documents = [f"{h}\n{b}" for h, b in sections],
)
```

No other file needs to change. The `faq_path` parameter is still passed by `EvalRunner` — you can ignore it once you have a persistent index.

---

## Persisting and Triaging Feedback

Traces are appended to `evals/layer3_feedback/storage/feedback_log.jsonl` (one JSON object per line) when `feedback.persist: true` in `eval_config.yaml`.

**Add `feedback_log.jsonl` to `.gitignore`** — it grows without bound and may contain PII from live queries.

```
# .gitignore
evals/layer3_feedback/storage/feedback_log.jsonl
```

**Weekly triage workflow:**

```bash
# 1. See what's in the log
python run_evals.py status

# 2. Cluster failures by root cause
python run_evals.py triage

# 3. Validate and promote
python run_evals.py promote --id <trace_id> --answer "<correct answer>"

# 4. Paste the printed YAML into golden_set/, open a PR
```

**Nine root-cause clusters** (and the team routed to each):

| Cluster | Team |
|---|---|
| `prompt_injection` | Security |
| `policy_exception_missed` | Compliance |
| `stale_knowledge` | Content Team |
| `poor_reasoning` | AI Team |
| `tool_failure` | Engineering |
| `bad_retrieval` | Engineering |
| `weak_instructions` | AI Team |
| `policy_ambiguity` | Compliance |
| `missing_context` | Engineering |

---

## Running Tests

```bash
# All tests (no API key required)
pytest tests/ -v

# Single layer
pytest tests/test_layer1.py -v
pytest tests/test_layer2.py -v
pytest tests/test_layer3.py -v

# Skip integration tests (default — same as CI)
pytest tests/ -m "not integration"

# Run integration tests (requires ANTHROPIC_API_KEY)
pytest tests/ -m integration
```

**86 tests, 0.2 s** — the full suite is fast because `MockClaudeJudge` is used and no disk I/O happens outside of the JSONL persistence tests (which use `tmp_path`).

---

## Token Usage

Every `EvalRunner.run()` call returns a `token_layers: list[LayerTokens]` you can inspect or render:

```python
from evals.token_tracker import render_token_table
print(render_token_table(result.token_layers))
```

```
  Token Usage──────────────────────────────────────────────────────
  Layer                             In      Out    Total  Type
  ---------------------------- -------  -------  -------  ------------
  Layer 1 - Ground Truth           262       31      293  no LLM call
  Layer 2a - Code Evals            232       28      260  no LLM call
  Layer 2b - LLM Judge           1,331      240    1,571  API call *
  Layer 3 - Feedback               234        0      234  no LLM call
  ──────────────────────────── ───────  ───────  ───────
  TOTAL                          2,059      299    2,358
```

Token counts use the `len(text) // 4` approximation (1 token ≈ 4 characters). Only Layer 2b (`LLM Judge`) generates real API cost; all other layers are local computation.

**Cost estimation** (Layer 2b, 4 dimensions, claude-sonnet-4-6 pricing):
- Input:  ~1,330 tokens × $3 / M tokens  ≈  $0.004 per eval run
- Output: ~240 tokens × $15 / M tokens   ≈  $0.004 per eval run
- **≈ $0.008 per golden entry evaluated with the real judge**

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for `ClaudeJudge`. Unset → `MockClaudeJudge`. |
| `USE_MOCK_JUDGE` | `false` | Set `true` to force `MockClaudeJudge` regardless of key. |
| `JUDGE_MODEL` | `claude-sonnet-4-6` | Anthropic model used by `ClaudeJudge`. |
| `JUDGE_MODEL_FAST` | `claude-haiku-4-5-20251001` | Reserved for a lighter judge pass. |
| `JUDGE_THRESHOLD` | `0.70` | Minimum weighted score for Layer 2b to pass. |
| `AUTO_PROMOTE_THRESHOLD` | `0.85` | Traces below this are auto-flagged for triage. |
