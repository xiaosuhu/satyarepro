import json

from satyarepro.types import ToolSchema

from ..base import Tool

_CELL_SEP = "\n\n# ── cell ──\n\n"


class NotebookParser(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="notebook_parser",
            description="Parse a Jupyter notebook (.ipynb) and return all code cells as Python source.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the .ipynb file.",
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            nb = json.load(fh)

        cells: list[str] = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                text = "".join(source) if isinstance(source, list) else source
                if text.strip():
                    cells.append(text)

        if not cells:
            return "# No code cells found in notebook."
        return _CELL_SEP.join(cells)
