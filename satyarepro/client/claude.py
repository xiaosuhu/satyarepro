import anthropic

from satyarepro.types import CompletionResponse, ToolCall, ToolSchema, Usage

from .base import ModelClient


class ClaudeClient(ModelClient):
    def __init__(self, model: str = "claude-sonnet-4-6", cache_system: bool = True) -> None:
        self._client = anthropic.AsyncAnthropic()
        self.model = model
        self.cache_system = cache_system

    def _build_system(self, system: str | None) -> list[dict] | str | None:
        if not system:
            return None
        if self.cache_system:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system

    def _parse_response(self, response: anthropic.types.Message) -> CompletionResponse:
        raw_content: list[dict] = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
                raw_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )

        return CompletionResponse(
            content=" ".join(text_parts),
            raw_content=raw_content,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=response.stop_reason,
        )

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        built = self._build_system(system)
        if built is not None:
            kwargs["system"] = built
        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            tools=anthropic_tools,
        )
        built = self._build_system(system)
        if built is not None:
            kwargs["system"] = built
        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)
