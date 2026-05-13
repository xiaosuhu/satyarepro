from dataclasses import dataclass, field

from satyarepro.client.base import ModelClient
from satyarepro.config import settings
from satyarepro.tools.base import ToolRegistry

_SYSTEM_PROMPT = """\
You are SatyaRepro, an expert AI agent for biomedical reproducibility auditing. \
Your task is to systematically evaluate whether a published biomedical study can be reproduced.

When auditing a paper, use available tools to:
1. Search PubMed to locate the paper and retrieve its abstract/details.
2. Extract and audit statistical claims (p-values, effect sizes, sample sizes).
3. Check for data and code availability statements.
4. Identify methodological concerns: blinding, randomisation, controls, power analysis.
5. Search for known replication attempts or critiques.

Conclude with a structured Reproducibility Assessment:
- Reproducibility Score: X/10
- Statistical Concerns: …
- Methodological Concerns: …
- Transparency Indicators: (code/data availability, pre-registration)
- Recommended Actions: …
"""


@dataclass
class AuditReport:
    summary: str
    tool_calls_made: int
    messages: list[dict] = field(default_factory=list)


class AuditOrchestrator:
    def __init__(
        self,
        client: ModelClient,
        tools: ToolRegistry,
        max_iterations: int | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_iterations = max_iterations or settings.max_audit_iterations

    async def audit(self, query: str) -> AuditReport:
        messages: list[dict] = [{"role": "user", "content": query}]
        tool_schemas = self.tools.schemas()
        tool_calls_made = 0
        last_content = ""

        for _ in range(self.max_iterations):
            response = await self.client.complete_with_tools(
                messages=messages,
                tools=tool_schemas,
                system=_SYSTEM_PROMPT,
            )
            last_content = response.content
            messages.append({"role": "assistant", "content": response.raw_content})

            if response.stop_reason != "tool_use" or not response.tool_calls:
                break

            tool_results = []
            for tc in response.tool_calls:
                try:
                    result = await self.tools.dispatch(tc.name, tc.input)
                except Exception as exc:
                    result = f"Tool error: {exc}"
                tool_calls_made += 1
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )

            messages.append({"role": "user", "content": tool_results})

        return AuditReport(
            summary=last_content,
            tool_calls_made=tool_calls_made,
            messages=messages,
        )
