from abc import ABC, abstractmethod

from satyarepro.types import CompletionResponse, ToolSchema


class ModelClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> CompletionResponse: ...

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> CompletionResponse: ...
