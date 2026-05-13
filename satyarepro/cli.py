"""Command-line interface for SatyaRepro."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


# ── static-only mode (Layer 1, no API key required) ──────────────────────────

async def _run_static(path: Path) -> None:
    from satyarepro.tools.layer1 import CheckpointCheck, DependencyCheck, SeedCheck, SplitCheck
    from satyarepro.tools.parsers import NotebookParser, ScriptParser

    parser = NotebookParser() if path.suffix == ".ipynb" else ScriptParser()
    code = await parser.execute(path=str(path))

    print(f"\n{'═' * 58}")
    print(f"  SatyaRepro — Static Audit")
    print(f"  File: {path}")
    print(f"{'═' * 58}\n")

    # Tools that operate on code
    for tool in [SeedCheck(), SplitCheck(), CheckpointCheck()]:
        result = json.loads(await tool.execute(code=code))
        _print_result(tool.schema.name, result)

    # DependencyCheck needs a requirements file
    reqs_path = path.parent / "requirements.txt"
    if reqs_path.exists():
        dep_result = json.loads(
            await DependencyCheck().execute(requirements=reqs_path.read_text())
        )
        _print_result("dependency_check", dep_result)
    else:
        _print_skip("dependency_check", f"no requirements.txt found in {path.parent}")

    print()


def _print_result(name: str, result: dict) -> None:
    status = result.get("status", "unknown")
    icon = "✓" if status == "ok" else ("✗" if status == "issues_found" else "?")
    print(f"  {icon}  {name}: {status}")
    if "error" in result:
        print(f"       error: {result['error']}")
    if "message" in result and status == "ok":
        print(f"       {result['message']}")
    for issue in result.get("issues", []):
        if isinstance(issue, dict):
            line = issue.get("line", "?")
            msg = issue.get("warning") or issue.get("call", str(issue))
            print(f"       line {line}: {msg}")
        else:
            print(f"       {issue}")
    for pkg in result.get("unpinned_packages", []):
        print(f"       unpinned: {pkg}")
    print()


def _print_skip(name: str, reason: str) -> None:
    print(f"  –  {name}: skipped ({reason})\n")


# ── full agentic mode ─────────────────────────────────────────────────────────

async def _run_full(path: Path, model: str, max_iter: int) -> None:
    from satyarepro.agent import AuditOrchestrator
    from satyarepro.client.claude import ClaudeClient
    from satyarepro.tools import create_default_registry

    client = ClaudeClient(model=model)
    registry = create_default_registry(client=client)
    orchestrator = AuditOrchestrator(client, registry, max_iterations=max_iter)

    query = (
        f"Run a full reproducibility audit on the file at this path: {path}\n\n"
        "Steps:\n"
        "1. Parse the file with the appropriate parser tool.\n"
        "2. Run seed_check, split_check, and checkpoint_check on the extracted code.\n"
        "3. Run leakage_detector, subgroup_reporter, and provenance_checker.\n"
        "4. Summarise all findings.\n"
        "5. Generate a TRIPOD-AI checklist and DMSP recommendations from the findings.\n"
        "Produce a structured final report."
    )

    print(f"\n{'═' * 58}")
    print(f"  SatyaRepro — Full Agentic Audit")
    print(f"  File:  {path}")
    print(f"  Model: {model}  |  Max iterations: {max_iter}")
    print(f"{'═' * 58}\n")
    print("  Running… (this may take a minute)\n")

    report = await orchestrator.audit(query)

    print(f"  Tool calls made: {report.tool_calls_made}")
    print(f"\n{'─' * 58}\n")
    print(report.summary)
    print(f"\n{'─' * 58}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="satyarepro",
        description="Audit a biomedical ML notebook or script for reproducibility issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  satyarepro test_notebook.ipynb --static-only\n"
            "  satyarepro experiment.py\n"
            "  satyarepro train.ipynb --model claude-opus-4-7 --max-iter 15"
        ),
    )
    p.add_argument("path", help="Path to a .ipynb or .py file.")
    p.add_argument(
        "--static-only",
        action="store_true",
        help="Run Layer 1 static analysis only — no API key required.",
    )
    p.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        metavar="MODEL",
        help="Claude model ID (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=10,
        metavar="N",
        help="Maximum agent iterations for full audit (default: 10).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    path = Path(args.path).resolve()

    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix not in {".ipynb", ".py"}:
        print(
            f"error: unsupported file type '{path.suffix}' — use .ipynb or .py",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.static_only:
        asyncio.run(_run_static(path))
    else:
        asyncio.run(_run_full(path, args.model, args.max_iter))


if __name__ == "__main__":
    main()
