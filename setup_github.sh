#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_github.sh — One-command setup to push llm-eval-stack to GitHub
#
# Prerequisites:
#   - git installed
#   - GitHub CLI installed: https://cli.github.com/  (brew install gh)
#   - OR: set GITHUB_TOKEN env var and we'll use the API directly
#
# Usage:
#   cd <folder containing this repo>
#   chmod +x setup_github.sh
#   ./setup_github.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO_NAME="llm-eval-stack"
REPO_DESC="Production-ready 3-layer eval stack for RAG Q&A: Ground Truth, Judgment Patterns, and Feedback Loops"

echo ""
echo "🔍 3-Layer Eval Stack — GitHub Setup"
echo "────────────────────────────────────"
echo ""

# ── Step 1: git init ──────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
  git init -b main
  echo "✅ Initialized git repo"
else
  echo "✅ Git repo already initialised"
fi

# ── Step 2: create .gitignore ─────────────────────────────────────────────────
cat > .gitignore << 'GITIGNORE'
.env
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
reports/*.html
evals/layer3_feedback/storage/feedback_log.jsonl
evals/layer2_judgment/human_review/review_queue.jsonl
GITIGNORE
echo "✅ Created .gitignore"

# ── Step 3: initial commit ────────────────────────────────────────────────────
git add -A
git commit -m "feat: initial 3-layer eval stack for RAG Q&A systems

Layer 1: Governed golden set — regulated/, historical-failures/, adversarial/
  - YAML entries with must_contain/must_not_contain hard gates
  - Examples: TILA disclosure, HIPAA PHI, backorder shipping, prompt injection

Layer 2: Three judgment patterns
  - Code evaluators (deterministic): eval_ssn_redacted, eval_apr_disclosed,
    eval_no_policy_override, eval_damaged_exception_honoured, and more
  - LLM-as-judge (Claude): faithfulness, relevance, completeness, correctness
  - Human review queue: priority-ordered, domain-routed, feeds Layer 3

Layer 3: Feedback loops (the compounding moat)
  - ProductionFeedbackCollector: weighted sampling toward failures
  - FailurePatternAnalyzer: 9 root-cause clusters with team routing
  - GoldenSetUpdater: failure → golden entry pipeline (same-week)

pytest test suite + GitHub Actions CI (eval-ci.yml)"
echo "✅ Initial commit created"

# ── Step 4: create GitHub repo ────────────────────────────────────────────────
if command -v gh &> /dev/null; then
  echo ""
  echo "Creating public GitHub repo via gh CLI..."
  gh repo create "$REPO_NAME" \
    --public \
    --description "$REPO_DESC" \
    --source=. \
    --remote=origin \
    --push
  echo ""
  echo "✅ Repo created and pushed!"
  echo "🔗 https://github.com/$(gh api user -q .login)/$REPO_NAME"
else
  # ── Fallback: manual remote setup ─────────────────────────────────────────
  echo ""
  echo "⚠️  GitHub CLI (gh) not found. Follow these steps to push manually:"
  echo ""
  echo "  1. Create a new EMPTY public repo at: https://github.com/new"
  echo "     Name it: $REPO_NAME"
  echo "     Do NOT initialise with README, .gitignore, or license."
  echo ""
  echo "  2. Run these commands (replace YOUR_USERNAME):"
  echo "     git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
  echo "     git push -u origin main"
  echo ""
  echo "  Or with SSH:"
  echo "     git remote add origin git@github.com:YOUR_USERNAME/$REPO_NAME.git"
  echo "     git push -u origin main"
fi

echo ""
echo "──────────────────────────────────────────────────────────"
echo "Next steps after pushing:"
echo ""
echo "  1. Add your ANTHROPIC_API_KEY as a GitHub secret:"
echo "     Settings → Secrets and variables → Actions → New secret"
echo "     Name: ANTHROPIC_API_KEY"
echo ""
echo "  2. Install dependencies locally:"
echo "     pip install -e '.[dev]'"
echo ""
echo "  3. Run the eval stack:"
echo "     pytest tests/ -m 'not integration'       # No API key needed"
echo "     python scripts/run_evals.py status        # Readiness report"
echo "     python scripts/run_evals.py run           # Layer 1 + 2 eval"
echo "     python scripts/run_evals.py triage        # Weekly failure triage"
echo ""
echo "  4. Add ANTHROPIC_API_KEY to .env and run LLM judge:"
echo "     python scripts/run_evals.py run --judge"
echo "──────────────────────────────────────────────────────────"
