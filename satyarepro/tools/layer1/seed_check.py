import ast
import json

from satyarepro.types import ToolSchema

from ..base import Tool
from .._utils import collect_calls, collect_imports

_SEED_CALLS: dict[str, set[str]] = {
    "numpy": {"numpy.random.seed"},
    "torch": {
        "torch.manual_seed",
        "torch.cuda.manual_seed",
        "torch.cuda.manual_seed_all",
    },
    "random": {"random.seed"},
    "tensorflow": {"tensorflow.random.set_seed", "tf.random.set_seed"},
}

_FRAMEWORK_LIBS: dict[str, set[str]] = {
    "numpy": {"numpy"},
    "torch": {"torch"},
    "random": {"random"},
    "tensorflow": {"tensorflow"},
}


class SeedCheck(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="seed_check",
            description=(
                "Detect missing random seed fixation in Python ML code. "
                "Checks numpy, PyTorch, Python random, and TensorFlow."
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

        aliases, libs = collect_imports(tree)
        calls = collect_calls(tree, aliases)

        issues: list[str] = []
        seeded: list[str] = []
        detected: list[str] = []

        for fw, required_libs in _FRAMEWORK_LIBS.items():
            if not (libs & required_libs):
                continue
            detected.append(fw)
            seed_funcs = _SEED_CALLS[fw]
            if any(c in seed_funcs for c in calls):
                seeded.append(fw)
            else:
                issues.append(
                    f"{fw}: seed not fixed (expected one of: {sorted(seed_funcs)})"
                )

        result: dict = {"frameworks_detected": detected, "seeded": seeded}
        if issues:
            result["status"] = "issues_found"
            result["issues"] = issues
        else:
            result["status"] = "ok"
        return json.dumps(result, indent=2)
