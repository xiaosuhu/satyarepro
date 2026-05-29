from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..base import Tool

_SYSTEM = (
    "You are a biomedical ML methodologist specialising in model training reproducibility. "
    "Be concise and specific: cite line numbers or variable names where possible."
)

_PROMPT_TEMPLATE = """\
Analyse the following Python biomedical ML code for hyperparameter reporting completeness.

Check specifically for:
1. Key hyperparameters hardcoded without documentation (learning rate, batch size, epochs, \
dropout rate, regularisation strength, etc.)
2. Presence of hyperparameter search (GridSearchCV, RandomizedSearchCV, Optuna, Ray Tune, \
Hyperopt, KerasTuner, etc.)
3. If search is present: whether the search space and best-found values are logged or reported
4. Whether the final hyperparameter values actually used are explicitly stated
5. Early stopping: whether stopping conditions are defined and the triggered epoch/iteration \
is recorded

For each issue: state the location, what is missing, severity (high/medium/low), and a concise fix.
If all hyperparameters are properly documented, confirm clearly.
Be concise: report only actionable findings, skip boilerplate.

```python
{code}
```"""


class HyperparameterReporter(Tool):
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
            name="hyperparameter_reporter",
            description=(
                "Check whether key hyperparameters are documented, whether any search was "
                "performed and its results recorded, and whether early stopping is defined."
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
