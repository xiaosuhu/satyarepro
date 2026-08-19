"""Tests for within_paper_robustness_checker (Replicability module).

Layer 1 checks (check_single_seed, check_no_multi_split) are pure AST scans,
tested directly with real code snippets — no mocking needed. Layer 2 checks
(check_variance_reported, check_significance_test) are tested with
MockClient for prompt routing and JSON parsing — no real LLM calls.
"""
from __future__ import annotations

import json

from satyarepro.client.mock import MockClient
from satyarepro.tools.layer2.replicability.within_paper_robustness_checker import (
    WithinPaperRobustnessChecker,
    check_no_multi_split,
    check_single_seed,
)
from satyarepro.types import CompletionResponse, Usage


def _ok_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        content=text,
        raw_content=[{"type": "text", "text": text}],
        usage=Usage(10, 5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — check_single_seed (pure AST scan, no mocks)
# ══════════════════════════════════════════════════════════════════════════════

_CODE_SINGLE_SEED = """\
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)
model = RandomForestClassifier(random_state=42)
"""

_CODE_MULTI_SEED_LOOP = """\
from sklearn.model_selection import train_test_split

results = []
for seed in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    results.append(train_model(X_train, y_train))
"""

_CODE_NO_SEED = """\
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
"""


class TestCheckSingleSeed:
    def test_single_hardcoded_seed_detected(self):
        result = check_single_seed(code=_CODE_SINGLE_SEED)
        assert result.check_id == "check_single_seed"
        assert result.layer == 1
        assert result.finding == "single_value_detected"

    def test_loop_over_multiple_seeds_detected(self):
        result = check_single_seed(code=_CODE_MULTI_SEED_LOOP)
        assert result.finding == "multiple_values_detected"

    def test_no_seed_usage_is_not_applicable(self):
        result = check_single_seed(code=_CODE_NO_SEED)
        assert result.finding == "not_applicable"

    def test_ignores_extra_kwargs_like_paper_text(self):
        # execute() forwards all kwargs to every check — code checks must
        # tolerate (and ignore) paper_text without error.
        result = check_single_seed(code=_CODE_SINGLE_SEED, paper_text="irrelevant")
        assert result.finding == "single_value_detected"

    def test_missing_code_is_not_applicable(self):
        result = check_single_seed(paper_text="no code here")
        assert result.finding == "not_applicable"

    def test_syntax_error_is_not_applicable_not_crash(self):
        result = check_single_seed(code="def broken(\n")
        assert result.finding == "not_applicable"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — check_no_multi_split (pure AST scan, no mocks)
# ══════════════════════════════════════════════════════════════════════════════

_CODE_SINGLE_SPLIT = """\
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""

_CODE_CROSS_VALIDATED = """\
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    pass
"""

_CODE_NO_SPLIT = _CODE_NO_SEED


class TestCheckNoMultiSplit:
    def test_single_split_detected(self):
        result = check_no_multi_split(code=_CODE_SINGLE_SPLIT)
        assert result.check_id == "check_no_multi_split"
        assert result.layer == 1
        assert result.finding == "single_value_detected"

    def test_cross_validation_detected(self):
        result = check_no_multi_split(code=_CODE_CROSS_VALIDATED)
        assert result.finding == "multiple_values_detected"

    def test_no_split_is_not_applicable(self):
        result = check_no_multi_split(code=_CODE_NO_SPLIT)
        assert result.finding == "not_applicable"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — check_variance_reported (MockClient, prompt routing + JSON parsing)
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckVarianceReported:
    async def test_reported_true_parses_finding(self):
        mock = MockClient()
        mock.enqueue(_ok_response('{"reported": true, "evidence": "AUC 0.85 \\u00b1 0.03", "confidence": "high"}'))
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_variance_reported(paper_text="We report AUC 0.85 ± 0.03 across 5 runs.")
        assert result.check_id == "check_variance_reported"
        assert result.layer == 2
        assert result.finding == "reported"
        assert result.confidence == "high"

    async def test_reported_false_parses_finding(self):
        mock = MockClient()
        mock.enqueue(_ok_response('{"reported": false, "evidence": "", "confidence": "medium"}'))
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_variance_reported(paper_text="Accuracy was 0.9.")
        assert result.finding == "not_reported"

    async def test_paper_text_forwarded_to_prompt(self):
        mock = MockClient()
        mock.enqueue(_ok_response('{"reported": false, "evidence": "", "confidence": "low"}'))
        tool = WithinPaperRobustnessChecker(client=mock)

        await tool.check_variance_reported(paper_text="UNIQUE_MARKER_TEXT_123")
        assert "UNIQUE_MARKER_TEXT_123" in mock.calls[0]["messages"][0]["content"]

    async def test_markdown_fenced_json_is_parsed(self):
        mock = MockClient()
        mock.enqueue(_ok_response('```json\n{"reported": true, "evidence": "CI reported", "confidence": "high"}\n```'))
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_variance_reported(paper_text="95% CI [0.80, 0.90]")
        assert result.finding == "reported"

    async def test_empty_paper_text_is_not_applicable_and_skips_llm(self):
        mock = MockClient()
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_variance_reported(paper_text="")
        assert result.finding == "not_applicable"
        assert mock.calls == []


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — check_significance_test (MockClient, prompt routing + JSON parsing)
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckSignificanceTest:
    async def test_tested_with_correction(self):
        mock = MockClient()
        mock.enqueue(_ok_response(
            '{"tested": true, "multiple_comparison_correction_mentioned": true, '
            '"evidence": "Bonferroni-corrected p-values", "confidence": "high"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_significance_test(paper_text="We applied Bonferroni correction.")
        assert result.finding == "tested_with_correction"

    async def test_tested_without_correction(self):
        mock = MockClient()
        mock.enqueue(_ok_response(
            '{"tested": true, "multiple_comparison_correction_mentioned": false, '
            '"evidence": "p < 0.05", "confidence": "medium"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_significance_test(paper_text="p < 0.05 for the main comparison.")
        assert result.finding == "tested_no_correction_mentioned"

    async def test_not_tested(self):
        mock = MockClient()
        mock.enqueue(_ok_response(
            '{"tested": false, "multiple_comparison_correction_mentioned": false, '
            '"evidence": "", "confidence": "medium"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_significance_test(paper_text="We report accuracy of 0.9.")
        assert result.finding == "not_tested"

    async def test_paper_text_forwarded_to_prompt(self):
        mock = MockClient()
        mock.enqueue(_ok_response(
            '{"tested": false, "multiple_comparison_correction_mentioned": false, '
            '"evidence": "", "confidence": "low"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        await tool.check_significance_test(paper_text="UNIQUE_MARKER_TEXT_456")
        assert "UNIQUE_MARKER_TEXT_456" in mock.calls[0]["messages"][0]["content"]

    async def test_empty_paper_text_is_not_applicable_and_skips_llm(self):
        mock = MockClient()
        tool = WithinPaperRobustnessChecker(client=mock)

        result = await tool.check_significance_test(paper_text="")
        assert result.finding == "not_applicable"
        assert mock.calls == []


# ══════════════════════════════════════════════════════════════════════════════
# Full tool — execute() end-to-end (MockClient only, no real LLM calls)
# ══════════════════════════════════════════════════════════════════════════════

class TestWithinPaperRobustnessCheckerExecute:
    async def test_schema_name_and_optional_fields(self):
        schema = WithinPaperRobustnessChecker().schema
        assert schema.name == "within_paper_robustness_checker"
        assert schema.input_schema["required"] == []
        assert set(schema.input_schema["properties"].keys()) == {"code", "paper_text"}

    async def test_execute_runs_all_four_checks_in_order(self):
        mock = MockClient()
        mock.enqueue(_ok_response('{"reported": false, "evidence": "", "confidence": "medium"}'))
        mock.enqueue(_ok_response(
            '{"tested": false, "multiple_comparison_correction_mentioned": false, '
            '"evidence": "", "confidence": "medium"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        raw = await tool.execute(code=_CODE_SINGLE_SEED, paper_text="This paper reports accuracy only.")
        result = json.loads(raw)

        check_ids = [c["check_id"] for c in result["checks_run"]]
        assert check_ids == [
            "check_single_seed",
            "check_no_multi_split",
            "check_variance_reported",
            "check_significance_test",
        ]

    async def test_execute_output_structure(self):
        mock = MockClient()
        mock.enqueue(_ok_response('{"reported": false, "evidence": "", "confidence": "medium"}'))
        mock.enqueue(_ok_response(
            '{"tested": false, "multiple_comparison_correction_mentioned": false, '
            '"evidence": "", "confidence": "medium"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        raw = await tool.execute(code=_CODE_SINGLE_SEED, paper_text="This paper reports accuracy only.")
        result = json.loads(raw)

        assert set(result.keys()) == {"checks_run", "summary", "caveats"}
        assert "static/textual" in result["caveats"]
        assert isinstance(result["summary"], str) and result["summary"]

    async def test_summary_lists_risk_signals_by_check_id(self):
        mock = MockClient()
        mock.enqueue(_ok_response('{"reported": false, "evidence": "", "confidence": "medium"}'))
        mock.enqueue(_ok_response(
            '{"tested": false, "multiple_comparison_correction_mentioned": false, '
            '"evidence": "", "confidence": "medium"}'
        ))
        tool = WithinPaperRobustnessChecker(client=mock)

        raw = await tool.execute(code=_CODE_SINGLE_SEED, paper_text="no stats reported at all")
        summary = json.loads(raw)["summary"]

        assert "check_single_seed" in summary
        assert "check_variance_reported" in summary
        assert "check_significance_test" in summary

    async def test_execute_with_no_input_is_all_not_applicable(self):
        mock = MockClient()
        tool = WithinPaperRobustnessChecker(client=mock)

        raw = await tool.execute()
        result = json.loads(raw)

        findings = [c["finding"] for c in result["checks_run"]]
        assert findings == ["not_applicable"] * 4
        assert mock.calls == []  # empty paper_text must never trigger an LLM call
        assert result["summary"] == "No robustness risk signals detected among the applicable checks."
