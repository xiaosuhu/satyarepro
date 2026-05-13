import asyncio
import json
import os
import tempfile
from pathlib import Path

from satyarepro.types import ToolSchema

from ..base import Tool

_PY_EXTENSIONS = {".py", ".ipynb"}
_MAX_FILE_CHARS = 8_000
_MAX_FILES = 20


def _extract_notebook_cells(path: Path) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
        cells = [
            "".join(c.get("source", []))
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        return "\n\n".join(cells)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


class RepoFetcher(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="repo_fetcher",
            description=(
                "Clone a public GitHub repository (shallow) and return the content of "
                "all Python (.py) and notebook (.ipynb) files, up to a reasonable size limit."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTPS URL of the GitHub repository."},
                    "max_files": {
                        "type": "integer",
                        "description": f"Maximum number of files to include (default {_MAX_FILES}).",
                        "default": _MAX_FILES,
                    },
                },
                "required": ["url"],
            },
        )

    async def execute(self, url: str, max_files: int = _MAX_FILES) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth=1", "--quiet", url, tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                return f"git clone failed: {stderr.decode(errors='replace').strip()}"

            py_files = sorted(
                [p for p in Path(tmpdir).rglob("*") if p.suffix in _PY_EXTENSIONS],
                key=lambda p: (len(p.parts), p.name),
            )[:max_files]

            if not py_files:
                return "No Python or notebook files found in the repository."

            sections: list[str] = []
            for fpath in py_files:
                rel = fpath.relative_to(tmpdir)
                try:
                    if fpath.suffix == ".ipynb":
                        content = _extract_notebook_cells(fpath)
                    else:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                    if len(content) > _MAX_FILE_CHARS:
                        content = content[:_MAX_FILE_CHARS] + f"\n# … truncated ({len(content)} chars total)"
                    sections.append(f"# === {rel} ===\n{content}")
                except Exception as exc:
                    sections.append(f"# === {rel} === [read error: {exc}]")

            return "\n\n".join(sections)
