from .notebook_parser import NotebookParser
from .script_parser import ScriptParser

_SUPPORTED = (".ipynb", ".py")


async def parse_input(path: str) -> str:
    """Route path to the appropriate parser and return code as a string.

    Raises ValueError for unsupported file types.
    """
    if path.endswith(".ipynb"):
        return await NotebookParser().execute(path)
    if path.endswith(".py"):
        return await ScriptParser().execute(path)
    raise ValueError(
        f"Unsupported file format: '{path}'. "
        f"Supported formats: {', '.join(_SUPPORTED)}"
    )
