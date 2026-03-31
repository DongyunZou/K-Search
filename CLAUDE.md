# CLAUDE.md — K-Search Development Guide

## Repository Overview

K-Search is an LLM-driven GPU kernel optimization system that maintains a **co-evolving world model** (a structured search tree of hypotheses) to guide iterative kernel generation. The main entry point is `generate_kernels_and_eval.py`; the world-model loop lives in `k_search/kernel_generators/kernel_generator_world_model.py`.

## Development Principles

- **Minimum diff**: make the smallest change that achieves the goal. Do not refactor surrounding code.
- **Feature flags first**: every new component must be gated behind a CLI flag (or config field) defaulting to `False`/disabled so it can be toggled for ablation studies.
- **No speculative abstractions**: add helpers only when they serve more than one concrete call site.
- **Use `uv run`**: this project uses [uv](https://github.com/astral-sh/uv) for Python environment management. Never invoke `python` directly — always prefix with `uv run`, e.g. `uv run python generate_kernels_and_eval.py ...`.

## Planned Features

### 1. Web Search for Agent (`--enable-web-search`)

Allow the coding/world-model LLM calls to optionally retrieve web search results (e.g. CUDA/Triton docs, recent papers) before generating a kernel.

- Gate: `--enable-web-search` flag (default off).
- Integration point: `KernelGenerator` (base class) — inject retrieved snippets into the system or user prompt before the LLM call.
- Keep search results bounded in length to avoid ballooning context.

### 2. PTX ISA Markdown Skill (`--enable-ptx-isa-skill`)

Read and index the PTX ISA documentation from <https://github.com/technillogue/ptx-isa-markdown> as a retrieval skill available to the coding model.

- Gate: `--enable-ptx-isa-skill` flag (default off).
- Implementation: fetch/clone the markdown files once (cache locally), chunk by section, provide a lightweight BM25 or embedding retrieval over them.
- Integration point: same prompt injection hook as web search (can share the same "context retrieval" abstraction).

### 3. NCU Profile Integration (`--enable-ncu-profile`)

Run NVIDIA Nsight Compute (`ncu`) profiling on FlashInfer-Bench kernels and feed the profile metrics back to the world model so it can revise action-node scores.

- Gate: `--enable-ncu-profile` flag (default off).
- Integration point: `flashinfer_bench_task.py` — after each `EvalResult`, optionally run `ncu` on the winning solution and attach metrics to the result.
- World model update: `WorldModelManager.refine(...)` should accept optional `ncu_metrics: dict` and include them in the refinement prompt so scores for hardware-bound actions (memory bandwidth, occupancy, …) are grounded in real data.

### 4. Separate World-Model and Coding Models (`--wm-model-name` / `--wm-base-url`)

Allow the world-model LLM calls (init, refine, action selection) to use a different model than the kernel-coding LLM calls.

- Gate: new CLI args `--wm-model-name` and `--wm-base-url` (default: same as `--model-name` / `--base-url`, i.e. no change when not set).
- Integration point: `generate_kernels_and_eval.py` — construct a second `OpenAI` client and pass it separately to `WorldModelKernelGeneratorWithBaseline` and `WorldModelManager`.
- The coding model continues to be controlled by `--model-name` / `--base-url`.

## Ablation Matrix

| Feature | Flag | Default |
|---------|------|---------|
| World model | `--world-model` | off |
| Web search | `--enable-web-search` | off |
| PTX ISA skill | `--enable-ptx-isa-skill` | off |
| NCU profiling | `--enable-ncu-profile` | off |
| Separate WM model | `--wm-model-name` (omit = share coding model) | off |

All flags are independent and combinable, so any subset can be enabled for a given experiment.

## Key Files

| File | Role |
|------|------|
| `generate_kernels_and_eval.py` | CLI entry point, constructs generator & runs loop |
| `k_search/kernel_generators/kernel_generator.py` | Base LLM generator (prompt → code) |
| `k_search/kernel_generators/kernel_generator_world_model.py` | World-model-aware generator |
| `k_search/kernel_generators/world_model_manager.py` | WM init / refine / select lifecycle |
| `k_search/kernel_generators/world_model.py` | WM data structures & JSON schema |
| `k_search/tasks/flashinfer_bench_task.py` | FlashInfer-Bench task adapter (NCU hook goes here) |
| `k_search/tasks/task_base.py` | `Solution`, `EvalResult` types |
