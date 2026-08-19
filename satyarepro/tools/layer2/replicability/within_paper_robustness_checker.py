from __future__ import annotations

import ast
import json
from typing import Any

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..._utils import dotted_name
from ...checklist import Check, CheckResult, ChecklistTool

# ── Layer 1 checks: static AST scanning, no LLM ──────────────────────────────

_SEED_NAME_HINTS = ("seed", "random_state")
_SPLIT_FUNC_NAME = "train_test_split"
_CV_CLASS_NAMES = {
    "KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold",
    "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit", "LeaveOneOut",
    "LeavePOut", "ShuffleSplit", "StratifiedShuffleSplit",
}
_CV_FUNC_NAMES = {"cross_val_score", "cross_validate", "cross_val_predict"}


def _is_seed_name(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in _SEED_NAME_HINTS)


def _loop_node_ids(tree: ast.AST) -> set[int]:
    """id()s of every node inside a for/while loop body, for quick membership checks."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            for sub in ast.walk(node):
                ids.add(id(sub))
    return ids


def check_single_seed(code: str = "", **_kwargs: Any) -> CheckResult:
    """Scan for seed/random_state assignments: one fixed value vs. multiple/looped values."""
    code = code or ""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CheckResult(
            check_id="check_single_seed",
            layer=1,
            finding="not_applicable",
            evidence=f"Could not parse code: {exc}",
            confidence="low",
        )

    loop_ids = _loop_node_ids(tree)
    literals: list[Any] = []
    in_loop = False
    usage_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_seed_name(target.id):
                    usage_count += 1
                    if id(node) in loop_ids:
                        in_loop = True
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, (int, str)
                    ):
                        literals.append(node.value.value)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and _is_seed_name(kw.arg):
                    usage_count += 1
                    if id(node) in loop_ids:
                        in_loop = True
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, (int, str)
                    ):
                        literals.append(kw.value.value)

    distinct_literals = sorted(set(literals), key=str)

    if in_loop or len(distinct_literals) >= 2:
        finding = "multiple_values_detected"
        evidence = (
            f"Seed/random_state usage found inside a loop or with {len(distinct_literals)} "
            f"distinct literal values: {distinct_literals}."
        )
        confidence = "medium"
    elif usage_count == 0:
        finding = "not_applicable"
        evidence = "No seed/random_state usage found in the provided code."
        confidence = "high"
    else:
        finding = "single_value_detected"
        shown = distinct_literals[0] if distinct_literals else "non-literal"
        evidence = (
            f"Found {usage_count} seed/random_state usage site(s), all using a single fixed "
            f"value ({shown}); no evidence of testing across multiple seeds."
        )
        confidence = "medium"

    return CheckResult(
        check_id="check_single_seed", layer=1, finding=finding, evidence=evidence, confidence=confidence
    )


def check_no_multi_split(code: str = "", **_kwargs: Any) -> CheckResult:
    """Scan for repeated/cross-validated splitting vs. a single one-shot train_test_split."""
    code = code or ""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CheckResult(
            check_id="check_no_multi_split",
            layer=1,
            finding="not_applicable",
            evidence=f"Could not parse code: {exc}",
            confidence="low",
        )

    loop_ids = _loop_node_ids(tree)
    split_call_count = 0
    split_in_loop = False
    cv_found = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        if not name:
            continue
        short = name.split(".")[-1]
        if short == _SPLIT_FUNC_NAME:
            split_call_count += 1
            if id(node) in loop_ids:
                split_in_loop = True
        elif short in _CV_CLASS_NAMES or short in _CV_FUNC_NAMES:
            cv_found = True

    if cv_found or split_in_loop or split_call_count >= 2:
        finding = "multiple_values_detected"
        evidence = (
            f"Cross-validation usage detected: {cv_found}; {split_call_count} "
            f"train_test_split call(s) found"
            f"{' inside a loop' if split_in_loop else ''}."
        )
        confidence = "medium"
    elif split_call_count == 1:
        finding = "single_value_detected"
        evidence = (
            "Exactly one train_test_split call found, no cross-validation or "
            "loop-based re-splitting detected."
        )
        confidence = "medium"
    else:
        finding = "not_applicable"
        evidence = "No train/test split or cross-validation calls found in the provided code."
        confidence = "high"

    return CheckResult(
        check_id="check_no_multi_split", layer=1, finding=finding, evidence=evidence, confidence=confidence
    )


# ── Layer 2 checks: LLM reasoning over paper text ────────────────────────────

_VARIANCE_SYSTEM = (
    "You are a biomedical ML methodologist checking whether a paper reports variance, "
    "confidence intervals, or standard deviation for its main results."
)
_VARIANCE_PROMPT = """\
Read the following paper text and determine whether it reports variance, confidence \
intervals, or standard deviation for its main results (e.g. "AUC 0.85 ± 0.03", \
"95% CI [0.80, 0.90]", standard deviation across repeated runs).

Paper text:
\"\"\"
{paper_text}
\"\"\"

Respond with ONLY a JSON object (no prose, no markdown fences) with these fields:
- "reported": true if variance/CI/std is explicitly reported for at least one main \
result, else false.
- "evidence": a short quote or paraphrase from the text supporting your answer, or "" \
if reported is false.
- "confidence": "high", "medium", or "low" — your confidence in this determination \
given the text provided."""

_SIGNIFICANCE_SYSTEM = (
    "You are a biomedical ML methodologist checking whether a paper performed statistical "
    "significance testing and, if so, whether it addressed multiple comparisons."
)
_SIGNIFICANCE_PROMPT = """\
Read the following paper text and determine two things:
1. Whether the authors performed a statistical significance test (e.g. t-test, DeLong \
test, McNemar's test, bootstrap confidence interval, permutation test) to support their \
main results.
2. If a significance test was performed, whether the paper mentions correction for \
multiple comparisons (e.g. Bonferroni, Benjamini-Hochberg/FDR) when applicable.

Paper text:
\"\"\"
{paper_text}
\"\"\"

Respond with ONLY a JSON object (no prose, no markdown fences) with these fields:
- "tested": true if a significance test was performed, else false.
- "multiple_comparison_correction_mentioned": true if multiple-comparison correction is \
mentioned, else false. Set to false if "tested" is false.
- "evidence": a short quote or paraphrase supporting your answer, or "" if tested is false.
- "confidence": "high", "medium", or "low" — your confidence in this determination given \
the text provided."""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find a JSON object in LLM response: {text!r}")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


# ── The tool ──────────────────────────────────────────────────────────────

_RISK_FINDINGS = frozenset(
    {
        "single_value_detected",  # check_single_seed / check_no_multi_split
        "not_reported",  # check_variance_reported
        "not_tested",  # check_significance_test
        "tested_no_correction_mentioned",  # check_significance_test
        "check_failed",  # base-class failure sentinel — surface it, don't hide it
    }
)


class WithinPaperRobustnessChecker(ChecklistTool):
    def __init__(self, client: ModelClient | None = None) -> None:
        self._client = client

    async def _get_client(self) -> ModelClient:
        if self._client is not None:
            return self._client
        from satyarepro.client.claude import ClaudeClient
        return ClaudeClient()

    @property
    def checks(self) -> list[Check]:
        return [
            check_single_seed,
            check_no_multi_split,
            self.check_variance_reported,
            self.check_significance_test,
        ]

    async def check_variance_reported(self, paper_text: str = "", **_kwargs: Any) -> CheckResult:
        paper_text = paper_text or ""
        if not paper_text.strip():
            return CheckResult(
                check_id="check_variance_reported",
                layer=2,
                finding="not_applicable",
                evidence="No paper_text provided.",
                confidence="high",
            )

        client = await self._get_client()
        response = await client.complete(
            messages=[{"role": "user", "content": _VARIANCE_PROMPT.format(paper_text=paper_text)}],
            system=_VARIANCE_SYSTEM,
            max_tokens=1024,
        )
        parsed = _extract_json_object(response.content)
        reported = bool(parsed.get("reported"))
        return CheckResult(
            check_id="check_variance_reported",
            layer=2,
            finding="reported" if reported else "not_reported",
            evidence=str(parsed.get("evidence", "")),
            confidence=str(parsed.get("confidence", "low")),
        )

    async def check_significance_test(self, paper_text: str = "", **_kwargs: Any) -> CheckResult:
        paper_text = paper_text or ""
        if not paper_text.strip():
            return CheckResult(
                check_id="check_significance_test",
                layer=2,
                finding="not_applicable",
                evidence="No paper_text provided.",
                confidence="high",
            )

        client = await self._get_client()
        response = await client.complete(
            messages=[{"role": "user", "content": _SIGNIFICANCE_PROMPT.format(paper_text=paper_text)}],
            system=_SIGNIFICANCE_SYSTEM,
            max_tokens=1024,
        )
        parsed = _extract_json_object(response.content)
        tested = bool(parsed.get("tested"))
        corrected = bool(parsed.get("multiple_comparison_correction_mentioned"))
        if not tested:
            finding = "not_tested"
        elif corrected:
            finding = "tested_with_correction"
        else:
            finding = "tested_no_correction_mentioned"
        return CheckResult(
            check_id="check_significance_test",
            layer=2,
            finding=finding,
            evidence=str(parsed.get("evidence", "")),
            confidence=str(parsed.get("confidence", "low")),
        )

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="within_paper_robustness_checker",
            description=(
                "Detect static/textual signals of within-paper statistical robustness: "
                "whether a single fixed random seed is used, whether train/test splitting "
                "is repeated or cross-validated, and whether the paper text reports "
                "variance/CI and significance testing. Does not re-run any code."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source code to analyse for seed/split patterns (optional)."
                        ),
                    },
                    "paper_text": {
                        "type": "string",
                        "description": (
                            "Paper text to analyse for variance/significance reporting (optional)."
                        ),
                    },
                },
                "required": [],
            },
        )

    def _summarize(self, results: list[CheckResult]) -> str:
        """Lists which specific checks show a robustness risk signal, by check_id.

        This tool's checks don't share the base class's "ok" literal — each has
        its own finding vocabulary — so risk is determined by an explicit set
        of "weak signal" finding values rather than a generic not-ok comparison.
        """
        flagged = [r for r in results if r.finding in _RISK_FINDINGS]
        if not flagged:
            return "No robustness risk signals detected among the applicable checks."
        names = ", ".join(f"{r.check_id} ({r.finding})" for r in flagged)
        return f"{len(flagged)}/{len(results)} checks show potential robustness risk signals: {names}."

    @property
    def caveats(self) -> str:
        return (
            "This is a static/textual signal detection only; it does not represent a final "
            "judgment and does not involve actually re-running the code to verify variance "
            "or robustness."
        )
