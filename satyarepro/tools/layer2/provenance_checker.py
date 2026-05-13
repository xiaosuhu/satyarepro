from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..base import Tool

_SYSTEM = (
    "You are a NIH research compliance expert specialising in Data Management "
    "and Sharing Plans (DMSP) and biomedical data provenance."
)

_PROMPT_TEMPLATE = """\
Analyse the following Python biomedical ML code and check whether data provenance is \
adequately described per NIH Data Management and Sharing Plan (DMSP) requirements.

Check for:
1. Dataset name and source clearly identified (variable names, comments, docstrings)
2. Data access procedures described (institutional approval, DUA, MOU references)
3. Data version or collection date specified
4. Patient population described (inclusion/exclusion criteria referenced)
5. Preprocessing and cleaning steps documented

For each requirement report: present (yes/partial/no), evidence from the code or comments, \
and specific recommendations.

```python
{code}
```"""


class ProvenanceChecker(Tool):
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
            name="provenance_checker",
            description=(
                "Check whether data source and provenance are described per NIH DMSP requirements."
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
