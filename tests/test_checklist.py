"""Tests for the ChecklistTool base layer (satyarepro/tools/checklist.py).

Skeleton-only: no business logic, just the run/collect/summarize/fail-safe
mechanics that concrete checklist-style tools will build on later.
"""
from __future__ import annotations

import asyncio
import json

from satyarepro.tools.checklist import ChecklistTool, CheckResult
from satyarepro.types import ToolSchema


def _sync_ok_check(**kwargs):
    return CheckResult(
        check_id="sync_ok", layer=1, finding="ok", evidence="nothing found", confidence="high"
    )


def _sync_issue_check(**kwargs):
    return CheckResult(
        check_id="sync_issue",
        layer=1,
        finding="missing_seed",
        evidence="no np.random.seed() call",
        confidence="medium",
    )


async def _async_ok_check(**kwargs):
    await asyncio.sleep(0)
    return CheckResult(
        check_id="async_ok", layer=2, finding="ok", evidence="LLM found nothing", confidence="high"
    )


def _raising_check(**kwargs):
    raise ValueError("boom")


async def _async_raising_check(**kwargs):
    raise RuntimeError("async boom")


def _bad_return_check(**kwargs):
    return "not a CheckResult"


class _DummyChecklistTool(ChecklistTool):
    def __init__(self, checks):
        self._checks = checks

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="dummy_checklist",
            description="test-only checklist tool",
            input_schema={"type": "object", "properties": {}},
        )

    @property
    def checks(self):
        return self._checks


# ══════════════════════════════════════════════════════════════════════════════
# Happy path — collecting sync + async results, default summary/caveats
# ══════════════════════════════════════════════════════════════════════════════

class TestChecklistToolHappyPath:
    async def test_collects_sync_and_async_results(self):
        tool = _DummyChecklistTool([_sync_ok_check, _async_ok_check])
        raw = await tool.execute()
        result = json.loads(raw)
        assert len(result["checks_run"]) == 2
        check_ids = {c["check_id"] for c in result["checks_run"]}
        assert check_ids == {"sync_ok", "async_ok"}

    async def test_default_summary_counts_non_ok_findings(self):
        tool = _DummyChecklistTool([_sync_ok_check, _sync_issue_check, _async_ok_check])
        raw = await tool.execute()
        result = json.loads(raw)
        assert result["summary"] == "1/3 checks flagged potential issues."

    async def test_default_caveats_is_empty_string(self):
        tool = _DummyChecklistTool([_sync_ok_check])
        raw = await tool.execute()
        result = json.loads(raw)
        assert result["caveats"] == ""

    async def test_kwargs_forwarded_to_each_check(self):
        seen = {}

        def _capturing_check(**kwargs):
            seen.update(kwargs)
            return CheckResult(check_id="capture", layer=1, finding="ok", evidence="", confidence="high")

        tool = _DummyChecklistTool([_capturing_check])
        await tool.execute(code="print(1)", extra=42)
        assert seen == {"code": "print(1)", "extra": 42}

    async def test_empty_checks_list(self):
        tool = _DummyChecklistTool([])
        raw = await tool.execute()
        result = json.loads(raw)
        assert result["checks_run"] == []
        assert result["summary"] == "0/0 checks flagged potential issues."

    async def test_check_result_fields_round_trip_through_json(self):
        tool = _DummyChecklistTool([_sync_issue_check])
        raw = await tool.execute()
        result = json.loads(raw)
        entry = result["checks_run"][0]
        assert entry == {
            "check_id": "sync_issue",
            "layer": 1,
            "finding": "missing_seed",
            "evidence": "no np.random.seed() call",
            "confidence": "medium",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Failure handling — one check's exception must not abort the run
# ══════════════════════════════════════════════════════════════════════════════

class TestChecklistToolFailureHandling:
    async def test_sync_check_exception_becomes_check_failed(self):
        tool = _DummyChecklistTool([_raising_check])
        raw = await tool.execute()
        failed = json.loads(raw)["checks_run"][0]
        assert failed["finding"] == "check_failed"
        assert failed["confidence"] == "low"
        assert "boom" in failed["evidence"]

    async def test_async_check_exception_becomes_check_failed(self):
        tool = _DummyChecklistTool([_async_raising_check])
        raw = await tool.execute()
        failed = json.loads(raw)["checks_run"][0]
        assert failed["finding"] == "check_failed"
        assert failed["confidence"] == "low"
        assert "async boom" in failed["evidence"]

    async def test_one_failing_check_does_not_stop_others(self):
        tool = _DummyChecklistTool([_sync_ok_check, _raising_check, _async_ok_check])
        raw = await tool.execute()
        findings = [c["finding"] for c in json.loads(raw)["checks_run"]]
        assert len(findings) == 3
        assert findings.count("check_failed") == 1
        assert findings.count("ok") == 2

    async def test_check_returning_wrong_type_is_treated_as_failure(self):
        tool = _DummyChecklistTool([_bad_return_check])
        raw = await tool.execute()
        assert json.loads(raw)["checks_run"][0]["finding"] == "check_failed"


# ══════════════════════════════════════════════════════════════════════════════
# Subclass overrides
# ══════════════════════════════════════════════════════════════════════════════

class TestSubclassOverrides:
    async def test_summarize_can_be_overridden(self):
        class _CustomSummaryTool(_DummyChecklistTool):
            def _summarize(self, results):
                return f"custom: {len(results)} ran"

        tool = _CustomSummaryTool([_sync_ok_check])
        raw = await tool.execute()
        assert json.loads(raw)["summary"] == "custom: 1 ran"

    async def test_caveats_can_be_overridden(self):
        class _CustomCaveatsTool(_DummyChecklistTool):
            @property
            def caveats(self):
                return "This is a feasibility-stage check."

        tool = _CustomCaveatsTool([_sync_ok_check])
        raw = await tool.execute()
        assert json.loads(raw)["caveats"] == "This is a feasibility-stage check."


# ══════════════════════════════════════════════════════════════════════════════
# Existing tools must be unaffected by this new optional layer
# ══════════════════════════════════════════════════════════════════════════════

class TestExistingToolsUnaffected:
    def test_leakage_detector_is_plain_tool_not_checklist_tool(self):
        from satyarepro.tools.base import Tool
        from satyarepro.tools.layer2.leakage_detector import LeakageDetector

        assert issubclass(LeakageDetector, Tool)
        assert not issubclass(LeakageDetector, ChecklistTool)

    def test_outcome_distribution_checker_is_plain_tool_not_checklist_tool(self):
        from satyarepro.tools.layer2.applicability.outcome_distribution_checker import (
            OutcomeDistributionChecker,
        )

        assert not issubclass(OutcomeDistributionChecker, ChecklistTool)
