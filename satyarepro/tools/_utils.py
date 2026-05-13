import ast


def dotted_name(node: ast.expr) -> str | None:
    """Reconstruct a dotted attribute name (e.g. 'np.random.seed') from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def collect_imports(tree: ast.AST) -> tuple[dict[str, str], set[str]]:
    """Return (alias_map, top_level_libs).

    alias_map maps each local name to its canonical top-level module
    (e.g. {'np': 'numpy', 'torch': 'torch'}).
    """
    aliases: dict[str, str] = {}
    libs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                local = alias.asname or top
                aliases[local] = top
                libs.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            libs.add(top)
    return aliases, libs


def collect_calls(tree: ast.AST, aliases: dict[str, str]) -> set[str]:
    """Return all normalized dotted call names, with aliases resolved."""
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name:
                parts = name.split(".")
                canonical = aliases.get(parts[0], parts[0])
                calls.add(".".join([canonical] + parts[1:]))
    return calls
