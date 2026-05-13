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
  (requirements.txt, setup.py, conda env)
- split_check: detect potential train/test data leakage patterns
- checkpoint_check: detect missing model checkpoint saving

### Layer 2 — Semantic Analysis (LLM-powered)
- leakage_detector: cross-function patient-level data leakage
- subgroup_reporter: check age/sex/race subgroup performance reporting
- provenance_checker: data source description vs NIH DMSP requirements

### Report Generators
- tripod_ai_generator: generate TRIPOD-AI checklist from audit results
- dmsp_generator: generate NIH Data Management & Sharing Plan draft

### Input Parsers (Year 1: code only)
- notebook_parser: parse .ipynb files
- script_parser: parse .py files
- repo_fetcher: clone GitHub repo, extract relevant files
# dataset_parser: to be added in Year 2

## Phasing
Year 1 (current): Python only, local deployment, 
                  TRIPOD-AI + NIH DMSP reports
Year 2: R, MATLAB support; cloud deployment; dataset_parser;
        OllamaClient
Year 3: DOME, CONSORT-AI standards

## Testing Strategy
- Layer 1 tools: unit tested with MockClient, no real LLM needed
- Layer 2 tools: tested with MockClient for structure,
  ClaudeClient for quality
- Each tool must be testable independently
- Benchmark dataset (annotated biomedical ML notebooks with 
  known issues) to be built separately in Year 1

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