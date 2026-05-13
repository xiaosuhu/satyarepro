import ast
import json

from satyarepro.types import ToolSchema

from ..base import Tool
from .._utils import collect_imports, dotted_name

_FIT_METHODS = {"fit", "fit_transform"}
_TRAIN_HINTS = {"train", "_train", "tr", "trn"}


def _looks_like_train_data(name: str) -> bool:
    lower = name.lower()
    return any(lower == h or lower.endswith("_" + h.lstrip("_")) or lower.startswith(h.lstrip("_")) for h in _TRAIN_HINTS)


class SplitCheck(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="split_check",
            description=(
                "Detect potential train/test data leakage patterns in Python ML code. "
                "Flags preprocessing fitted on the full dataset instead of training data only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to analyse."},
                },
                "required": ["code"],
            },
        )

    async def execute(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return json.dumps({"error": f"Syntax error: {exc}"})

        issues: list[dict] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            short = func.attr
            if short not in _FIT_METHODS:
                continue
            if not node.args:
                continue
            arg_name = dotted_name(node.args[0])
            if arg_name and not _looks_like_train_data(arg_name):
                method = dotted_name(func) or f"<obj>.{short}"
                issues.append(
                    {
                        "line": getattr(node, "lineno", "?"),
                        "call": f"{method}({arg_name})",
                        "warning": (
                            f"'{arg_name}' does not appear to be training-only data. "
                            "Fitting on the full dataset leaks test-set statistics."
                        ),
                    }
                )

        if not issues:
            return json.dumps({"status": "ok", "message": "No obvious split-leakage patterns detected."})
        return json.dumps({"status": "issues_found", "issues": issues}, indent=2)
