import ast
import json

from satyarepro.types import ToolSchema

from ..base import Tool
from .._utils import collect_calls, collect_imports

_TRAINING_CALLS = {
    "model.fit",
    "trainer.train",
    "optimizer.step",
    "loss.backward",
}
_TRAINING_ATTRS = {"backward", "step"}

_SAVE_CALLS = {
    "torch.save",
    "model.save",
    "model.save_weights",
    "model.save_pretrained",
    "joblib.dump",
    "pickle.dump",
    "tf.saved_model.save",
    "tf.keras.models.save_model",
}
_SAVE_ATTRS = {"save", "save_weights", "save_pretrained", "dump"}


def _has_attr_call(tree: ast.AST, attrs: set[str]) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in attrs:
                return True
    return False


class CheckpointCheck(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="checkpoint_check",
            description=(
                "Detect missing model checkpoint saving in Python ML code. "
                "Flags training loops that do not persist the trained model."
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

        aliases, _ = collect_imports(tree)
        calls = collect_calls(tree, aliases)

        training_found = bool(calls & _TRAINING_CALLS) or _has_attr_call(tree, _TRAINING_ATTRS)
        if not training_found:
            return json.dumps(
                {"status": "ok", "message": "No training code detected — checkpoint check not applicable."}
            )

        save_found = bool(calls & _SAVE_CALLS) or _has_attr_call(tree, _SAVE_ATTRS)
        if save_found:
            return json.dumps({"status": "ok", "message": "Model checkpoint saving detected."})

        return json.dumps(
            {
                "status": "issues_found",
                "issues": ["Training loop detected but no model checkpoint saving found."],
                "recommendation": (
                    "Add model saving (e.g. torch.save, model.save, joblib.dump) "
                    "so the trained model can be reproduced without retraining."
                ),
            },
            indent=2,
        )
