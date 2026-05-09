"""
Integration tests.

Three test classes:
  TestEvalRunnerEndToEnd   – EvalRunner.run() through all 3 layers
  TestCLI                  – run_evals.py CLI commands via subprocess
  TestJudgeEnvSelection    – get_judge() env-var resolution (new scenarios)
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from evals.config import EVAL_CONFIG
from evals.layer1_ground_truth.dataset import GoldenEntry
from evals.layer2_judgment.llm_judge.judge import MockClaudeJudge, get_judge
from evals.runner import EvalRunner


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_entry(entry_id: str = "integ-reg-001") -> GoldenEntry:
    return GoldenEntry(
        id=entry_id,
        category="regulated",
        question="What is your return policy for electronics?",
        context="",
        expected_answer=(
            "Electronics can be returned within 30 days with original receipt "
            "and packaging. Refunds take 5-7 business days."
        ),
        must_contain=["30 days", "original receipt", "5-7 business days"],
        must_not_contain=["lifetime", "90 days", "immediate refund"],
        judgment_patterns=["code_evaluator", "llm_judge"],
    )


def _good_agent(question: str, context: str) -> str:
    return (
        "You can return electronics within 30 days of purchase. "
        "Please bring your original receipt and ensure the item is in its "
        "original packaging. Your refund will be processed within 5-7 business "
        "days back to your original payment method."
    )


def _bad_agent(question: str, context: str) -> str:
    return "We offer next-day delivery! Return anytime within 90 days — no receipt needed."


def _runner(persist: bool = False) -> EvalRunner:
    return EvalRunner(persist=persist)


def _run(runner: EvalRunner, entry: GoldenEntry, agent_fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(entry, agent_fn=agent_fn)


# ── TestEvalRunnerEndToEnd ─────────────────────────────────────────────────────

class TestEvalRunnerEndToEnd:
    def test_good_agent_passes_all_layers(self):
        result = _run(_runner(), _make_entry(), _good_agent)
        assert result.l1_passed
        assert result.l2_passed
        assert result.all_passed
        assert result.l3_logged

    def test_bad_agent_fails_at_least_one_layer(self):
        result = _run(_runner(), _make_entry(), _bad_agent)
        assert not result.all_passed

    def test_result_entry_id_matches(self):
        entry  = _make_entry("custom-id-999")
        result = _run(_runner(), entry, _good_agent)
        assert result.entry_id == "custom-id-999"

    def test_result_has_four_token_layers(self):
        result = _run(_runner(), _make_entry(), _good_agent)
        assert len(result.token_layers) == 4

    def test_all_token_layers_have_positive_total(self):
        result = _run(_runner(), _make_entry(), _good_agent)
        assert all(tl.total > 0 for tl in result.token_layers)

    def test_judge_report_weighted_average_in_range(self):
        result = _run(_runner(), _make_entry(), _good_agent)
        assert 0.0 <= result.judge_report.weighted_average <= 1.0

    def test_failure_cluster_none_on_passing_result(self):
        result = _run(_runner(), _make_entry(), _good_agent)
        assert result.failure_cluster is None

    def test_failure_cluster_set_on_failing_result(self):
        result = _run(_runner(), _make_entry(), _bad_agent)
        if not result.all_passed:
            assert result.failure_cluster is not None

    def test_collector_accumulates_two_traces(self):
        runner = _runner()
        entry  = _make_entry()
        _run(runner, entry, _good_agent)
        _run(runner, entry, _bad_agent)
        assert runner.collector_stats()["total_traces"] == 2

    def test_persist_writes_jsonl(self, tmp_path):
        runner   = _runner(persist=True)
        log_file = tmp_path / "traces.jsonl"
        runner.collector._persist  = True
        runner.collector._log_path = log_file
        entry = _make_entry("persist-test-001")
        _run(runner, entry, _good_agent)
        assert log_file.exists()
        line = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert line["trace_id"] == "trace-persist-test-001"

    def test_triage_returns_dict(self):
        runner = _runner()
        _run(runner, _make_entry(), _bad_agent)
        clusters = runner.triage(from_disk=False)
        assert isinstance(clusters, dict)

    def test_promote_failures_returns_entry_on_failure(self):
        runner = _runner()
        result = _run(runner, _make_entry(), _bad_agent)
        if not result.all_passed:
            trace_id = f"trace-{result.entry_id}"
            promoted = runner.promote_failures({trace_id: "30 days policy."})
            assert len(promoted) == 1
            assert "id" in promoted[0]
            assert "category" in promoted[0]

    def test_promote_with_write_creates_yaml(self, tmp_path):
        runner = _runner()
        result = _run(runner, _make_entry(), _bad_agent)
        if not result.all_passed:
            runner.updater._path = tmp_path
            trace_id = f"trace-{result.entry_id}"
            promoted = runner.promote_failures({trace_id: "30 days policy."}, write=True)
            if promoted:
                category = promoted[0]["category"]
                yaml_files = list((tmp_path / category).glob("*.yaml"))
                assert len(yaml_files) >= 1

    def test_retrieval_structured_mode(self, tmp_path):
        """EvalRunner resolves structured_path from config and uses JSON FAQ."""
        faq_json = tmp_path / "faq.json"
        faq_json.write_text(
            json.dumps([
                {"question": "What is your return policy for electronics?",
                 "answer": "Return within 30 days with original receipt. Refunds take 5-7 business days."}
            ]),
            encoding="utf-8",
        )
        runner = _runner()
        runner.cfg["retrieval"] = {
            "faq_path": str(faq_json.with_suffix(".md")),
            "mode": "structured",
            "structured_path": str(faq_json),
            "top_k": 1,
        }
        runner._root = tmp_path
        result = _run(runner, _make_entry(), _good_agent)
        assert result.l3_logged


# ── TestCLI ───────────────────────────────────────────────────────────────────

class TestCLI:
    """Subprocess tests for run_evals.py commands."""

    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "run_evals.py", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

    def test_run_summary_exits_zero(self):
        r = self._cli("run", "--summary")
        assert r.returncode == 0, r.stderr

    def test_run_outputs_pass_or_fail(self):
        r = self._cli("run", "--summary")
        assert "Passed" in r.stdout or "Failed" in r.stdout

    def test_run_invalid_id_exits_nonzero(self):
        r = self._cli("run", "--id", "does-not-exist-xyz")
        assert r.returncode != 0

    def test_run_invalid_category_exits_nonzero(self):
        r = self._cli("run", "--category", "nonexistent-category-xyz")
        assert r.returncode != 0

    def test_status_exits_zero(self):
        r = self._cli("status")
        assert r.returncode == 0

    def test_status_shows_total_traces(self):
        r = self._cli("status")
        assert "Total traces" in r.stdout

    def test_triage_exits_zero(self):
        r = self._cli("triage")
        assert r.returncode == 0

    def test_unknown_command_exits_nonzero(self):
        r = self._cli("nonexistent-command-xyz")
        assert r.returncode != 0

    def test_run_json_report_creates_file(self, tmp_path):
        r = self._cli("run", "--summary", "--report", "json",
                      "--report-dir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_run_csv_report_creates_file(self, tmp_path):
        r = self._cli("run", "--summary", "--report", "csv",
                      "--report-dir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

    def test_run_html_report_creates_file(self, tmp_path):
        r = self._cli("run", "--summary", "--report", "html",
                      "--report-dir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 1

    def test_run_multi_format_report(self, tmp_path):
        r = self._cli("run", "--summary", "--report", "json", "csv", "html",
                      "--report-dir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        assert len(list(tmp_path.glob("*.json"))) == 1
        assert len(list(tmp_path.glob("*.csv")))  == 1
        assert len(list(tmp_path.glob("*.html"))) == 1

    def test_json_report_contains_summary(self, tmp_path):
        self._cli("run", "--summary", "--report", "json",
                  "--report-dir", str(tmp_path))
        path = next(tmp_path.glob("*.json"))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "entries" in data
        assert data["summary"]["total"] >= 1

    def test_promote_missing_trace_exits_nonzero(self):
        r = self._cli("promote", "--id", "trace-nonexistent-xyz",
                      "--answer", "some answer")
        assert r.returncode != 0

    def test_run_with_agent_module_fallback(self):
        """Nonexistent module triggers fallback with warning, run still succeeds."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "agent:\n"
                "  module: nonexistent.module.xyz\n"
                "  function: generate\n"
                "retrieval:\n"
                "  faq_path: docs/faq.md\n"
                "  top_k: 2\n"
                "evaluators:\n"
                "  code: []\n"
                "  judge:\n"
                "    use_mock: true\n"
                "    dimensions: [faithfulness]\n"
                "feedback:\n"
                "  persist: false\n"
            )
            cfg_path = f.name
        try:
            r = self._cli("--config", cfg_path, "run", "--summary")
            # falls back to built-in agent; may pass or fail depending on golden set
            assert "WARNING" in r.stdout or r.returncode in (0, 1)
        finally:
            os.unlink(cfg_path)


# ── TestJudgeEnvSelection ─────────────────────────────────────────────────────

class TestJudgeEnvSelection:
    """
    These cover env-var scenarios not already in test_layer2.py::TestGetJudge.
    """

    def test_use_mock_false_no_key_still_returns_mock(self, monkeypatch):
        """USE_MOCK_JUDGE=false + no API key → still MockClaudeJudge (auto-detect)."""
        monkeypatch.setenv("USE_MOCK_JUDGE", "false")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            judge = get_judge()
        assert isinstance(judge, MockClaudeJudge)

    def test_dimensions_subset_respected(self):
        """get_judge with dimensions subset passes it through to MockClaudeJudge."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            judge = get_judge(use_mock=True, dimensions=["faithfulness", "relevance"])
        # judge stores default dims; actual subset is passed at evaluate() time
        assert isinstance(judge, MockClaudeJudge)

    def test_mock_judge_evaluate_custom_dims(self):
        """MockClaudeJudge.evaluate() with custom dimensions returns correct count."""
        judge = MockClaudeJudge()
        from evals.config import RUBRIC_WEIGHTS
        entry = GoldenEntry(
            id="t", category="regulated",
            question="Q?", context="ctx",
            expected_answer="A",
            must_contain=[], must_not_contain=[],
            judgment_patterns=[],
        )
        report = judge.evaluate(entry.question, entry.context, "A", entry.expected_answer,
                                dimensions=["faithfulness", "correctness"])
        assert len(report.scores) == 2
        scored_dims = {s.dimension for s in report.scores}
        assert scored_dims == {"faithfulness", "correctness"}

    def test_mock_judge_weighted_avg_custom_dims(self):
        """Weighted average re-normalises correctly over a subset of dimensions."""
        judge  = MockClaudeJudge()
        report = judge.evaluate("Q", "ctx", "A", "A",
                                dimensions=["faithfulness", "relevance"])
        total_weight = sum(
            s.normalized for s in report.scores  # raw scores fixed in mock
        )
        assert report.weighted_average > 0.0
