import ast

from satyarepro.types import ToolSchema

from ..base import Tool
from .notebook_parser import _strip_magic

_SCRIPT_HEADER = "# ── script ──\n\n"


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
            source = fh.read()

        cleaned = _strip_magic(source)
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            cleaned = f"# [script skipped — syntax error: {exc.msg} (line {exc.lineno})]\n"

        return _SCRIPT_HEADER + cleaned
