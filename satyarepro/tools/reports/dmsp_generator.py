from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..base import Tool

_SYSTEM = (
    "You are a NIH grant specialist with deep expertise in Data Management and Sharing Plans. "
    "Generate compliant, actionable DMSP drafts."
)

_PROMPT_TEMPLATE = """\
Based on the following biomedical AI audit findings, generate a draft NIH Data Management \
and Sharing Plan (DMSP) that addresses identified provenance and transparency gaps.

The DMSP must cover these six elements (per NIH NOT-OD-21-013):
1. Data Type — scientific data, metadata, and associated files to be preserved/shared
2. Related Tools and Software — required to access or manipulate the data
3. Standards — metadata standards, data formats, and terminology used
4. Data Preservation, Access, and Timeline — repository selection, timeline, persistent identifiers
5. Access, Distribution, and Reuse — who can access, under what conditions, and when
6. Oversight — roles and responsibilities for data management

Use [PLACEHOLDER] for information not available in the audit. Mark sections needing \
special attention based on gaps found in the audit with ⚠.

Audit Findings:
{audit_results}"""


class DMSPGenerator(Tool):
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
            name="dmsp_generator",
            description=(
                "Generate a draft NIH Data Management and Sharing Plan (DMSP) "
                "from collected audit findings."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "audit_results": {
                        "type": "string",
                        "description": "JSON or text summary of all prior audit tool findings.",
                    },
                },
                "required": ["audit_results"],
            },
        )

    async def execute(self, audit_results: str) -> str:
        client = await self._get_client()
        response = await client.complete(
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(audit_results=audit_results)}],
            system=_SYSTEM,
            max_tokens=4096,
        )
        return response.content
