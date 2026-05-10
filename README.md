# K-Search

LLM-driven GPU kernel optimization with a co-evolving world model.

## Environment setup

This project uses [uv](https://github.com/astral-sh/uv). Always invoke Python via `uv run`.

### 1. Clone flashinfer-bench (latest main)

The PyPI release of `flashinfer-bench` is missing `DsaTopkIndexerEvaluator`, which causes numerical validation failures on overflow workloads. Install the latest main from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench-main
```

### 2. Sync the environment

```bash
uv sync
```

This creates `.venv/` and installs `torch==2.12.0+cu132` (from the PyTorch test channel), `flashinfer-bench` in editable mode, plus the rest of the project dependencies.

### 3. Install DeepGEMM (required for the DSA topk indexer baseline)

`deep_gemm` is not on PyPI. Clone it with submodules and install in editable mode (build depends on torch, so `--no-build-isolation` is required):

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
uv pip install -e /tmp/DeepGEMM --no-build-isolation
```

> `uv sync` removes packages not in the lockfile, so re-run the `uv pip install` above after every `uv sync`.

### 4. Verify

```bash
uv run python -c "
import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
import deep_gemm; print('deep_gemm: OK')
from flashinfer_bench.bench.evaluators.dsa_topk_indexer import DsaTopkIndexerEvaluator; print('DsaTopkIndexerEvaluator: OK')
"
```

### 5. Configure LLM credentials

Create a `.env` file in the repo root (already gitignored):

```
LLM_API_KEY=<your OpenAI-compatible api key>
LLM_BASE_URL=<your provider's base URL, e.g. https://api.openai.com/v1>
```

CLI flags `--api-key` / `--base-url` override `.env` if provided.

## Dataset

```bash
mkdir -p ~/dataset/flashinfer-trace
uv run hf download flashinfer-ai/mlsys26-contest --repo-type=dataset --local-dir ~/dataset/flashinfer-trace/
```
