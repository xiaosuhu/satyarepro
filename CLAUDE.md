# SatyaRepro — Developer Guide for Claude Code

## What is SatyaRepro
An agentic AI tool that audits biomedical AI/ML workflows for 
reproducibility issues and generates standards-compliant reports.
Connected to DAIR3 (NIH grant 5R25GM151182-03) at University of 
Michigan MIDAS.

## Input / Output
Input: Jupyter notebook (.ipynb), Python script (.py), 
       Git repository (URL)
Output 1: Reproducibility Audit Report (specific issues + fixes)
Output 2: Standards-Compliant Report (TRIPOD-AI, NIH DMSP)

## Architecture
Central LLM Agent orchestrates all analysis via tool calls.
The agent decides which tools to call, in what order, and 
how many times — not a fixed pipeline.

Two categories of tools:
- Layer 1: Static analysis tools (Python AST) — deterministic,
  no LLM inference needed, called by agent
- Layer 2: Semantic analysis tools — require LLM reasoning,
  agent uses these for complex contextual checks

ModelClient abstraction layer: agent's "brain" is swappable.
Backends: ClaudeClient, MockClient (testing), 
OllamaClient (local, to be added in Year 2).

## Tool Library
These are the tools available to the LLM agent via function 
calling. The agent decides which tools to invoke and in what 
order based on the input code.

### Layer 1 — Static Analysis (no LLM)
- seed_check: detect missing random seed fixation 
  (numpy, torch, random, tensorflow)
- dependency_check: detect missing version pins 
  (requirements.txt or environment.yml passed as content string)
- split_check: detect potential train/test data leakage patterns
  (handles chained calls e.g. StandardScaler().fit_transform(X))
- checkpoint_check: detect missing model checkpoint saving

### Layer 2 — Semantic Analysis (LLM-powered)
- leakage_detector: cross-function patient-level data leakage
- subgroup_reporter: check age/sex/race subgroup performance reporting
- provenance_checker: data source description vs NIH DMSP requirements

### Report Generators
- tripod_ai_generator: generate TRIPOD-AI checklist from audit results
- dmsp_generator: generate NIH Data Management & Sharing Plan draft

### Input Parsers (Year 1: code only)
- notebook_parser: parse .ipynb files — strips Jupyter magic commands
  (%matplotlib, !pip, etc.) and skips cells with syntax errors; 
  remaining cells are always valid Python for downstream AST tools
- script_parser: parse .py files
- repo_fetcher: clone GitHub repo (shallow), extract relevant files
# dataset_parser: to be added in Year 2

## File Structure
satyarepro/
├── types.py          — shared dataclasses (ToolCall, ToolSchema, 
│                        CompletionResponse, Usage)
├── config.py         — pydantic-settings (.env, ANTHROPIC_API_KEY,
│                        CLAUDE_MODEL, NCBI_EMAIL, MAX_AUDIT_ITERATIONS)
├── cli.py            — CLI entry point (satyarepro command)
├── client/           — ModelClient abstraction
│   ├── base.py       — abstract ModelClient
│   ├── claude.py     — ClaudeClient (prompt caching on system prompt)
│   └── mock.py       — MockClient (FIFO queue, for tests)
├── agent/
│   └── orchestrator.py — AuditOrchestrator (tool-use loop)
├── api/              — FastAPI app
│   ├── app.py        — create_app() factory
│   ├── schemas.py    — Pydantic request/response models
│   └── routers/
│       ├── audit.py  — POST /audit (202 + background task),
│       │               GET /audit/{id} (poll status)
│       └── health.py — GET /health
└── tools/
    ├── base.py       — Tool ABC, ToolRegistry
    ├── _utils.py     — shared AST helpers (dotted_name,
    │                    collect_imports, collect_calls)
    ├── layer1/       — seed_check, dependency_check,
    │                    split_check, checkpoint_check
    ├── layer2/       — leakage_detector, subgroup_reporter,
    │                    provenance_checker
    ├── reports/      — tripod_ai_generator, dmsp_generator
    └── parsers/      — notebook_parser, script_parser, repo_fetcher

## CLI Usage
# Install (editable):
pip install -e .

# Static audit — no API key required:
satyarepro notebook.ipynb --static-only

# Full agentic audit — requires ANTHROPIC_API_KEY:
satyarepro notebook.ipynb
satyarepro script.py --model claude-opus-4-7 --max-iter 15

# Also runnable as a module:
python -m satyarepro.cli notebook.ipynb --static-only

## Phasing
Year 1 (current): Python only, local deployment, 
                  TRIPOD-AI + NIH DMSP reports
Year 2: R, MATLAB support; cloud deployment; dataset_parser;
        OllamaClient
Year 3: DOME, CONSORT-AI standards

## Implementation Status (Year 1)
Built and tested (68 tests passing):
- All 12 tools implemented and registered in create_default_registry()
- Layer 2 tools and report generators accept optional ModelClient;
  fall back to lazy ClaudeClient() if none passed
- ClaudeClient caches the system prompt (cache_control: ephemeral)
  to reduce token cost across agent iterations
- CLI supports --static-only (no API key) and full agentic mode
- FastAPI server (main.py / uvicorn) for HTTP access

Known gaps for Year 1 completion:
- dependency_check requires requirements.txt content passed separately;
  CLI --static-only auto-discovers requirements.txt in the same dir
- Benchmark dataset (annotated notebooks with known issues) not yet built
- OllamaClient not yet implemented (Year 2)

## Testing Strategy
- Layer 1 tools: unit tested deterministically — no LLM or MockClient needed
- Layer 2 tools + report generators: tested with MockClient for structure
  and prompt routing; ClaudeClient for quality (manual / integration)
- Parsers: tested with tmp_path fixtures (no network)
- Each tool is independently testable via ToolRegistry.dispatch()
- Run tests: python -m pytest tests/ -q

## Deployment
- Local: Ollama + open-weight model (Gemma 4) for 
  privacy-sensitive data (Year 2)
- Cloud: Claude API / Gemini API for broader accessibility
- Config controls which backend is active

## Do Not Change
- client/ directory (ModelClient abstraction is correct)
- agent/orchestrator.py (agent loop structure is correct)
- api/ directory (FastAPI structure is correct)
- tests/ structure (keep existing test patterns)
- tools/base.py interface (Tool ABC and ToolRegistry contract)
