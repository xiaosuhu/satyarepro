import json
import re

from satyarepro.types import ToolSchema

from ..base import Tool

_VERSION_OPS = ("==", ">=", "<=", "~=", "!=", ">", "<")


def _parse_requirements_txt(content: str) -> list[str]:
    unpinned: list[str] = []
    for line in content.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        if not any(op in line for op in _VERSION_OPS):
            pkg = re.split(r"[\[\s;]", line)[0].strip()
            if pkg:
                unpinned.append(pkg)
    return unpinned


def _parse_conda_env(content: str) -> list[str]:
    unpinned: list[str] = []
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "dependencies:":
            in_deps = True
            continue
        if not in_deps:
            continue
        if stripped.startswith("- ") and not stripped.startswith("- pip:"):
            pkg = stripped[2:].strip()
            if pkg and "=" not in pkg and not pkg.startswith("{"):
                unpinned.append(pkg)
        elif stripped and not stripped.startswith("-") and stripped.endswith(":"):
            in_deps = False
    return unpinned


class DependencyCheck(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="dependency_check",
            description=(
                "Check for missing version pins in a dependency file "
                "(requirements.txt or environment.yml). "
                "Unpinned packages are a reproducibility risk."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "string",
                        "description": "Content of the dependency file.",
                    },
                    "file_type": {
                        "type": "string",
                        "description": 'File type: "requirements.txt" (default) or "environment.yml".',
                        "default": "requirements.txt",
                    },
                },
                "required": ["requirements"],
            },
        )

    async def execute(self, requirements: str, file_type: str = "requirements.txt") -> str:
        if file_type == "environment.yml":
            unpinned = _parse_conda_env(requirements)
        else:
            unpinned = _parse_requirements_txt(requirements)

        if not unpinned:
            return json.dumps({"status": "ok", "message": "All packages have version pins."})
        return json.dumps(
            {
                "status": "issues_found",
                "unpinned_packages": unpinned,
                "count": len(unpinned),
                "recommendation": (
                    "Pin all packages to exact versions (==) for reproducibility."
                ),
            },
            indent=2,
        )
