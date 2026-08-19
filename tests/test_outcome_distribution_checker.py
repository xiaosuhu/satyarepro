"""Tests for outcome_distribution_checker (Applicability module).

Stage 2 (IQR / judgment) is pure Python: tested directly with fixed numbers,
no mocking needed. Stage 1 (Asta search -> LLM extraction) is tested with a
MockClient for prompt routing / JSON parsing and a stub AstaClient — no real
network or LLM calls.

corpus_id (Semantic Scholar's stable identifier) is the only reliable field
used for candidate exclusion; Asta's snippet_search does not reliably return
a DOI, so doi is reference-only and must never affect exclusion.
"""
from __future__ import annotations

import json

import pytest

from satyarepro.client.mock import MockClient
from satyarepro.tools.layer2.applicability.outcome_distribution_checker import (
    OutcomeDistributionChecker,
    _extract_json_array,
    _format_snippets,
    _format_summary,
    _judge,
)
from satyarepro.types import CompletionResponse, Usage


def _ok_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        content=text,
        raw_content=[{"type": "text", "text": text}],
        usage=Usage(10, 5),
    )


class FakeAstaClient:
    """Duck-typed stand-in for AstaClient — no MCP/network involved."""

    def __init__(self, snippet_results: list[dict] | None = None) -> None:
        self._snippet_results = snippet_results or []
        self.search_calls: list[dict] = []

    async def snippet_search(self, query: str, limit: int = 10, **extra):
        self.search_calls.append({"query": query, "limit": limit, **extra})
        return self._snippet_results

    async def get_citations(self, doi: str, limit: int = 20, **extra):
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — pure Python IQR / judgment logic (fixed numbers, no mocks)
# ══════════════════════════════════════════════════════════════════════════════

class TestJudge:
    def _candidate(self, value, same_task=True, corpus_id="1", doi="10.1/a", title="Study"):
        return {
            "title": title,
            "corpus_id": corpus_id,
            "doi": doi,
            "extracted_value": value,
            "sample_size": 100,
            "same_task": same_task,
            "justification": "same input modality and outcome",
        }

    def test_insufficient_data_below_min_comparators(self):
        candidates = [self._candidate(0.8, corpus_id=str(i)) for i in range(3)]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.9,
            candidates=candidates,
            exclude_corpus_id=None,
            min_comparators=4,
        )
        assert result["judgment"] == "insufficient_data"
        assert result["computed_range"] is None
        assert len(result["comparators"]) == 3

    def test_within_range(self):
        # Values 0.70, 0.72, 0.75, 0.78, 0.80 -> target 0.76 is comfortably inside IQR.
        values = [0.70, 0.72, 0.75, 0.78, 0.80]
        candidates = [self._candidate(v, corpus_id=str(i)) for i, v in enumerate(values)]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.76,
            candidates=candidates,
            exclude_corpus_id=None,
            min_comparators=4,
        )
        assert result["judgment"] == "within_range"
        assert result["computed_range"]["n"] == 5

    def test_outside_range(self):
        # Tight cluster around 0.70-0.74; target 0.99 is a clear outlier.
        values = [0.70, 0.71, 0.72, 0.73, 0.74]
        candidates = [self._candidate(v, corpus_id=str(i)) for i, v in enumerate(values)]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.99,
            candidates=candidates,
            exclude_corpus_id=None,
            min_comparators=4,
        )
        assert result["judgment"] == "outside_range"
        rng = result["computed_range"]
        assert 0.99 > rng["upper_fence"]

    def test_different_task_excluded_from_comparators(self):
        candidates = [
            self._candidate(0.7, corpus_id="1"),
            self._candidate(0.7, corpus_id="2", same_task=False),
        ]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.7,
            candidates=candidates,
            exclude_corpus_id=None,
            min_comparators=1,
        )
        assert len(result["comparators"]) == 1
        assert len(result["excluded_candidates"]) == 1
        assert result["excluded_candidates"][0]["exclusion_reason"] == "different_task"

    def test_exclude_corpus_id_filtered_out(self):
        candidates = [
            self._candidate(0.7, corpus_id="111"),
            self._candidate(0.7, corpus_id="222"),
        ]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.7,
            candidates=candidates,
            exclude_corpus_id="  111  ",  # whitespace-insensitive match
            min_comparators=1,
        )
        assert len(result["comparators"]) == 1
        assert result["comparators"][0]["corpus_id"] == "222"
        assert result["excluded_candidates"][0]["exclusion_reason"] == "excluded_corpus_id"

    def test_doi_never_used_for_exclusion(self):
        # One candidate has no DOI at all; the other's DOI happens to contain
        # the excluded corpus_id as a substring/lookalike. Neither should
        # affect exclusion — only corpus_id is checked.
        candidates = [
            self._candidate(0.7, corpus_id="111", doi=None),
            self._candidate(0.7, corpus_id="222", doi="10.1/111"),
        ]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.7,
            candidates=candidates,
            exclude_corpus_id="111",
            min_comparators=1,
        )
        comparator_ids = [c["corpus_id"] for c in result["comparators"]]
        assert comparator_ids == ["222"]
        assert result["excluded_candidates"][0]["corpus_id"] == "111"
        assert result["excluded_candidates"][0]["exclusion_reason"] == "excluded_corpus_id"

    def test_null_extracted_values_dont_crash_and_stay_insufficient(self):
        candidates = [
            self._candidate(None, corpus_id="1"),
            self._candidate(None, corpus_id="2"),
            self._candidate(0.8, corpus_id="3"),
            self._candidate(0.8, corpus_id="4"),
        ]
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.8,
            candidates=candidates,
            exclude_corpus_id=None,
            min_comparators=4,
        )
        # 4 same_task comparators clears min_comparators, but only 2 have numeric values.
        assert len(result["comparators"]) == 4
        assert result["computed_range"]["n"] == 2

    def test_caveat_always_present(self):
        result = _judge(
            query="q",
            target_metric_name="AUC",
            target_metric_value=0.8,
            candidates=[],
            exclude_corpus_id=None,
            min_comparators=4,
        )
        assert "feasibility-stage" in result["caveats"]


# ══════════════════════════════════════════════════════════════════════════════
# JSON extraction from LLM text
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractJsonArray:
    def test_raw_json_array(self):
        text = '[{"title": "A", "same_task": true}]'
        parsed = _extract_json_array(text)
        assert parsed == [{"title": "A", "same_task": True}]

    def test_markdown_fenced_json(self):
        text = '```json\n[{"title": "A", "same_task": false}]\n```'
        parsed = _extract_json_array(text)
        assert parsed[0]["title"] == "A"

    def test_no_array_raises(self):
        with pytest.raises(ValueError):
            _extract_json_array("no json here at all")

    def test_malformed_json_between_brackets_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json_array("[1, 2,]")  # trailing comma is invalid JSON


# ══════════════════════════════════════════════════════════════════════════════
# Full tool — MockClient (LLM) + FakeAstaClient (search), no real network
# ══════════════════════════════════════════════════════════════════════════════

_CANDIDATES_JSON = json.dumps(
    [
        {
            "title": "ECG mortality model A",
            "corpus_id": "1001",
            "doi": "10.1/a",
            "extracted_value": 0.81,
            "sample_size": 500,
            "same_task": True,
            "justification": "same modality and outcome",
        },
        {
            "title": "ECG mortality model B",
            "corpus_id": "1002",
            "doi": None,  # DOI unreliable/absent — must not block use as a comparator
            "extracted_value": 0.79,
            "sample_size": 300,
            "same_task": True,
            "justification": "same modality and outcome",
        },
        {
            "title": "Unrelated diabetes study",
            "corpus_id": "1003",
            "doi": "10.1/c",
            "extracted_value": None,
            "sample_size": None,
            "same_task": False,
            "justification": "different predicted outcome",
        },
    ]
)


class TestOutcomeDistributionChecker:
    async def test_schema_name(self):
        assert OutcomeDistributionChecker().schema.name == "outcome_distribution_checker"

    async def test_asta_query_built_from_task_description(self):
        mock = MockClient()
        mock.enqueue(_ok_response("[]"))
        asta = FakeAstaClient()
        tool = OutcomeDistributionChecker(client=mock, asta_client=asta)

        await tool.execute(
            target_metric_name="AUC-ROC",
            target_metric_value=0.9,
            task_description="12-lead ECG to predict 1-year mortality",
            min_comparators=1,
        )

        assert len(asta.search_calls) == 1
        assert "12-lead ECG to predict 1-year mortality" in asta.search_calls[0]["query"]
        assert "AUC-ROC" in asta.search_calls[0]["query"]

    async def test_snippets_forwarded_to_llm_prompt(self):
        mock = MockClient()
        mock.enqueue(_ok_response("[]"))
        # Real Asta shape: nested paper/snippet, corpusId as the identifier.
        asta = FakeAstaClient(
            snippet_results=[
                {
                    "paper": {"corpusId": "999", "title": "Some Paper", "doi": "10.1/x"},
                    "snippet": {"text": "reports AUC 0.8"},
                }
            ]
        )
        tool = OutcomeDistributionChecker(client=mock, asta_client=asta)

        await tool.execute(
            target_metric_name="AUC-ROC",
            target_metric_value=0.9,
            task_description="task X",
            min_comparators=1,
        )

        prompt = mock.calls[0]["messages"][0]["content"]
        assert "999" in prompt
        assert "Some Paper" in prompt
        assert "reports AUC 0.8" in prompt

    async def test_end_to_end_insufficient_data_below_min(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_CANDIDATES_JSON))
        asta = FakeAstaClient(snippet_results=[{"title": "x", "corpus_id": "0"}])
        tool = OutcomeDistributionChecker(client=mock, asta_client=asta)

        raw = await tool.execute(
            target_metric_name="AUC-ROC",
            target_metric_value=0.95,
            task_description="12-lead ECG to predict 1-year mortality",
            min_comparators=4,
        )
        result = json.loads(raw)
        assert result["judgment"] == "insufficient_data"
        assert len(result["comparators"]) == 2  # only same_task == true
        assert len(result["excluded_candidates"]) == 1

    async def test_end_to_end_respects_exclude_corpus_id(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_CANDIDATES_JSON))
        asta = FakeAstaClient(snippet_results=[{"title": "x", "corpus_id": "0"}])
        tool = OutcomeDistributionChecker(client=mock, asta_client=asta)

        raw = await tool.execute(
            target_metric_name="AUC-ROC",
            target_metric_value=0.95,
            task_description="12-lead ECG to predict 1-year mortality",
            exclude_corpus_id="1001",
            min_comparators=1,
        )
        result = json.loads(raw)
        corpus_ids = [c["corpus_id"] for c in result["comparators"]]
        assert "1001" not in corpus_ids
        assert "1002" in corpus_ids

    async def test_output_has_expected_top_level_keys(self):
        mock = MockClient()
        mock.enqueue(_ok_response("[]"))
        tool = OutcomeDistributionChecker(client=mock, asta_client=FakeAstaClient())

        raw = await tool.execute(
            target_metric_name="AUC-ROC",
            target_metric_value=0.9,
            task_description="task X",
        )
        result = json.loads(raw)
        assert set(result.keys()) == {
            "query",
            "target_metric",
            "comparators",
            "excluded_candidates",
            "computed_range",
            "judgment",
            "caveats",
        }


# ══════════════════════════════════════════════════════════════════════════════
# _format_summary — quick human-readable summary, not a report generator
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatSummary:
    def test_summarizes_insufficient_data(self):
        result = {
            "query": "task X AUC-ROC performance",
            "target_metric": {"name": "AUC-ROC", "value": 0.9},
            "comparators": [],
            "excluded_candidates": [],
            "computed_range": None,
            "judgment": "insufficient_data",
            "caveats": "feasibility-stage disclaimer",
        }
        summary = _format_summary(json.dumps(result))
        assert "AUC-ROC = 0.9" in summary
        assert "insufficient_data" in summary
        assert len(summary.splitlines()) <= 6

    def test_summarizes_outside_range(self):
        result = {
            "query": "q",
            "target_metric": {"name": "AUC-ROC", "value": 0.99},
            "comparators": [{}, {}, {}, {}],
            "excluded_candidates": [{}],
            "computed_range": {
                "n": 4, "values": [0.7, 0.71, 0.72, 0.73],
                "q1": 0.7075, "q3": 0.7225, "iqr": 0.015,
                "lower_fence": 0.685, "upper_fence": 0.7375,
            },
            "judgment": "outside_range",
            "caveats": "feasibility-stage disclaimer",
        }
        summary = _format_summary(json.dumps(result))
        assert "outside_range" in summary
        assert "n=4" in summary


class TestFormatSnippets:
    def test_empty_snippets(self):
        assert "no search results" in _format_snippets([])

    def test_formats_flat_shape(self):
        block = _format_snippets([{"title": "T", "corpus_id": "123", "doi": "D", "snippet": "S"}])
        assert "T" in block and "123" in block and "D" in block and "S" in block

    def test_formats_real_nested_asta_shape(self):
        block = _format_snippets(
            [{"paper": {"corpusId": "456", "title": "T2"}, "snippet": {"text": "S2"}}]
        )
        assert "456" in block and "T2" in block and "S2" in block
