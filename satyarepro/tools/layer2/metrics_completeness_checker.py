from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..base import Tool

_SYSTEM = (
    "You are a biomedical ML statistician specialising in clinical model reporting completeness. "
    "Be concise and specific: cite line numbers or variable names where possible."
)

_PROMPT_TEMPLATE = """\
Analyse the following Python biomedical ML code for evaluation metrics completeness.

Check specifically for:
1. Sole reliance on accuracy (misleading on imbalanced clinical datasets)
2. Missing metrics: AUC-ROC, F1, precision, recall, sensitivity, specificity, PPV, NPV
3. Absence of confidence intervals or standard deviations on reported metrics
4. Absence of statistical significance tests (e.g. DeLong test for AUC comparison, \
McNemar, bootstrap)
5. Whether evaluation is performed on an independent held-out test set (vs. only \
validation/cross-validation)
6. Whether calibration is assessed (Brier score, calibration curve, Hosmer-Lemeshow test)

For each issue: state the location, what is missing, severity (high/medium/low), and a concise fix.
If the metric suite is complete, confirm clearly.
Be concise: report only actionable findings, skip boilerplate.

```python
{code}
```"""


class MetricsCompletenessChecker(Tool):
    def __init__(self, client: ModelClient | None = None) -> None:
        self._client = client

    async def _get_client(self) -> ModelClient:
        if self._client is not None:
            return self._client
        from satyarepro.client.claude import ClaudeClient
        return ClaudeClient()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="metrics_completeness_checker",
            description=(
                "Check whether the evaluation metric suite is complete for a clinical ML model: "
                "AUC, F1, calibration, confidence intervals, and independent test set evaluation."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to analyse."},
                },
                "required": ["code"],
            },
        )

    async def execute(self, code: str) -> str:
        client = await self._get_client()
        response = await client.complete(
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(code=code)}],
            system=_SYSTEM,
            max_tokens=4096,
        )
        return response.content
