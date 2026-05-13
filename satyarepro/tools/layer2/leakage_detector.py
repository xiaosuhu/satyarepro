from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..base import Tool

_SYSTEM = (
    "You are a biomedical ML security expert specialising in data leakage detection. "
    "Be concise and specific: cite line numbers or function names where possible."
)

_PROMPT_TEMPLATE = """\
Analyse the following Python biomedical ML code for cross-function patient-level data leakage.

Look specifically for:
1. Patient/sample IDs present in both training and test sets without stratification
2. Test-subject data leaking into training through shared preprocessing state
3. Target-variable leakage (features derived from the label)
4. Temporal leakage (future data used to predict past events)
5. Any non-causal feature–target correlation paths

For each issue: state the location (function/line ref), leakage type, severity (high/medium/low), and a concise fix.
If no leakage is found, confirm clearly.

```python
{code}
```"""


class LeakageDetector(Tool):
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
            name="leakage_detector",
            description=(
                "Use LLM reasoning to detect cross-function patient-level data leakage "
                "that static analysis cannot catch."
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
            max_tokens=2048,
        )
        return response.content
