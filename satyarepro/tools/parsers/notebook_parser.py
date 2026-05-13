import ast
import json

from satyarepro.types import ToolSchema

from ..base import Tool

_CELL_SEP = "\n\n# ── cell ──\n\n"
_MAGIC_PREFIXES = ("%", "!")


def _strip_magic(source: str) -> str:
    """Replace Jupyter magic/shell lines with comments so ast.parse() can proceed."""
    lines = []
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith(_MAGIC_PREFIXES):
            lines.append(f"# [magic] {line.rstrip()}\n")
        else:
            lines.append(line)
    return "".join(lines)


def _prepare_cell(source: str, cell_num: int) -> tuple[str, str | None]:
    """Return (text_to_include, error_message).

    Strips magic lines then validates syntax. On failure returns a
    placeholder comment instead of the broken source so downstream
    ast.parse() calls on the full notebook still succeed.
    """
    cleaned = _strip_magic(source)
    try:
        ast.parse(cleaned)
        return cleaned, None
    except SyntaxError as exc:
        placeholder = f"# [cell {cell_num} skipped — syntax error: {exc.msg} (line {exc.lineno})]\n"
        return placeholder, f"cell {cell_num}: {exc.msg} (line {exc.lineno})"


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
        skipped: list[str] = []
        cell_num = 0

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            cell_num += 1
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else source
            if not text.strip():
                continue
            prepared, error = _prepare_cell(text, cell_num)
            cells.append(prepared)
            if error:
                skipped.append(error)

        if not cells:
            return "# No code cells found in notebook."

        header = ""
        if skipped:
            header = (
                "# NOTE: the following cells were skipped due to syntax errors:\n"
                + "".join(f"#   {s}\n" for s in skipped)
                + "\n"
            )
        return header + _CELL_SEP.join(cells)
