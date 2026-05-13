from satyarepro.types import CompletionResponse, ToolSchema, Usage

from .base import ModelClient


class MockClient(ModelClient):
    """In-memory client that returns pre-queued responses. Designed for tests."""

    def __init__(self, responses: list[CompletionResponse] | None = None) -> None:
        self._queue: list[CompletionResponse] = list(responses or [])
        self.calls: list[dict] = []

    def enqueue(self, response: CompletionResponse) -> None:
        self._queue.append(response)

    def _default_response(self) -> CompletionResponse:
        text = "Mock audit complete."
        return CompletionResponse(
            content=text,
            raw_content=[{"type": "text", "text": text}],
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        self.calls.append({"type": "complete", "messages": messages, "system": system})
        return self._queue.pop(0) if self._queue else self._default_response()

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> CompletionResponse:
        self.calls.append({"type": "complete_with_tools", "messages": messages})
        return self._queue.pop(0) if self._queue else self._default_response()
