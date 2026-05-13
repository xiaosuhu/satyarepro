from satyarepro.types import ToolSchema

from ..base import Tool


class ScriptParser(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="script_parser",
            description="Read a Python script (.py) and return its source code.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the .py file.",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
