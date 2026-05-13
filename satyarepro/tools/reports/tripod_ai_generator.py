from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ..base import Tool

_SYSTEM = (
    "You are a biomedical AI reporting standards expert specialising in TRIPOD-AI compliance. "
    "Generate structured, actionable checklists."
)

_PROMPT_TEMPLATE = """\
Based on the following biomedical AI audit findings, generate a TRIPOD-AI compliance checklist.

TRIPOD-AI (Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis \
or Diagnosis — Artificial Intelligence) requires reporting across these sections:
- Title/Abstract (items 1–3)
- Introduction: Objectives (items 4–5)
- Methods: Source of data, Participants, Outcomes, Predictors, Sample size, Missing data, \
  Statistical analysis methods, Development/validation (items 6–16)
- Results: Participants, Model development, Model performance (items 17–20)
- Discussion: Limitations, Interpretation, Implications (items 21–24)
- Other: Supplementary info, Funding (items 25–26)

For each section indicate: ✓ Reported / △ Partial / ✗ Not Reported / N/A.
Cite specific evidence from the audit findings. Flag the three highest-priority gaps.

Audit Findings:
{audit_results}"""


class TripodAIGenerator(Tool):
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
            name="tripod_ai_generator",
            description=(
                "Generate a TRIPOD-AI compliance checklist from collected audit findings."
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
