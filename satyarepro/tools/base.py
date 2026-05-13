from abc import ABC, abstractmethod
from typing import Any

from satyarepro.types import ToolSchema


class Tool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, *tools: Tool) -> "ToolRegistry":
        for tool in tools:
            self._tools[tool.schema.name] = tool
        return self

    def schemas(self) -> list[ToolSchema]:
        return [t.schema for t in self._tools.values()]

    async def dispatch(self, name: str, inputs: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name!r}")
        return await tool.execute(**inputs)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
