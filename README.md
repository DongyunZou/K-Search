# K-Search

LLM-driven GPU kernel optimization with a co-evolving world model.

`generate_kernels_and_eval.py` is the entry point. Per-kernel launch scripts under
`scripts/` wrap it with curated workloads + sensible defaults. Everything is env-var
driven so the same scripts work locally and as enterprise batch jobs.

---

## 1. Environment setup

This project uses [uv](https://github.com/astral-sh/uv). Always invoke Python via `uv run`.

### 1.1 Clone flashinfer-bench (latest main)

The PyPI release of `flashinfer-bench` is missing `DsaTopkIndexerEvaluator`. Install latest main:

```bash
git clone https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench-main
```

`pyproject.toml` declares `flashinfer-bench = { path = "/tmp/flashinfer-bench-main", editable = true }`. If you want it somewhere else, edit `pyproject.toml` (or override the path with `tool.uv.sources` before `uv sync`).

### 1.2 Sync the environment

```bash
uv sync
```

Creates `.venv/` and installs `torch==2.12.0+cu132` (PyTorch test channel), `flashinfer-bench` editable, plus the rest.

### 1.3 Install DeepGEMM (required for DSA topk-indexer baseline)

`deep_gemm` is not on PyPI. Clone **with submodules** and install editable (build depends on torch, so `--no-build-isolation` is required):

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
uv pip install -e /tmp/DeepGEMM --no-build-isolation
```

DeepGEMM JIT-compiles paged-MQA-logits kernels at runtime and looks for CUTLASS headers under `deep_gemm/include/{cutlass,cute}`. Editable install skips the build step that copies them, so symlink to the CUTLASS shipped with `flashinfer`:

```bash
FI_CUTLASS=$(uv run python -c "import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).parent/'data'/'cutlass'/'include')")
ln -sfn "$FI_CUTLASS/cutlass" /tmp/DeepGEMM/deep_gemm/include/cutlass
ln -sfn "$FI_CUTLASS/cute"    /tmp/DeepGEMM/deep_gemm/include/cute
```

> `uv sync` removes packages not in the lockfile, so re-run the `uv pip install` above after every `uv sync`. The symlinks survive `uv sync` (they live inside `/tmp/DeepGEMM`) but break if `flashinfer` reinstalls into a different venv path.

### 1.4 Sanity check

```bash
uv run python -c "
import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
import deep_gemm; print('deep_gemm: OK')
from flashinfer_bench.bench.evaluators.dsa_topk_indexer import DsaTopkIndexerEvaluator; print('DsaTopkIndexerEvaluator: OK')
"
```

---

## 2. Dataset

```bash
mkdir -p ~/dataset/flashinfer-trace
uv run hf download flashinfer-ai/mlsys26-contest --repo-type=dataset --local-dir ~/dataset/flashinfer-trace/
```

### 2.1 Patch the DSA topk-indexer baseline (required)

The shipped baseline `solutions/baseline/dsa/dsa_topk_indexer_*/flashinfer_deepgemm_wrapper_2ba145.json` was written against an older DeepGEMM API. Current DeepGEMM `get_paged_mqa_logits_metadata` / `fp8_paged_mqa_logits` require `context_lens` to be **2-D `[batch, next_n]`**, but the baseline passes the 1-D `seq_lens`. Without patching, the baseline returns `RUNTIME_ERROR` on every workload.

Apply this one-time patch (preserves semantics, only reshapes for deep_gemm):

```bash
uv run python - <<'PY'
import json
from pathlib import Path
p = Path.home() / "dataset/flashinfer-trace/solutions/baseline/dsa/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/flashinfer_deepgemm_wrapper_2ba145.json"
d = json.loads(p.read_text())
mainpy = next(sf for sf in d["sources"] if sf["path"] == "main.py")
src = mainpy["content"]
old1 = "    num_sms = torch.cuda.get_device_properties(device).multi_processor_count\n    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, page_size, num_sms)"
new1 = "    num_sms = torch.cuda.get_device_properties(device).multi_processor_count\n    seq_lens_2d = seq_lens.view(-1, 1).contiguous()\n    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seq_lens_2d, page_size, num_sms)"
if old1 in src:
    src = src.replace(old1, new1, 1)
    src = src.replace("        weights,\n        seq_lens,\n        block_table,", "        weights,\n        seq_lens_2d,\n        block_table,", 1)
    mainpy["content"] = src
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    print("patched")
else:
    print("already patched or marker not found")
PY
```

### 2.2 Verify baselines compile and pass

End-to-end smoke test for the DSA family (you can adapt for gdn / moe similarly):

```bash
uv run python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages

dataset_root = Path.home() / "dataset/flashinfer-trace"
manifest = json.loads(Path("scripts/selected_workloads.json").read_text())

for family, key in [("dsa", "dsa_sparse_attention"), ("dsa", "dsa_topk_indexer")]:
    info = manifest[key]; defn = info["definition"]
    sol_dir = dataset_root / "solutions" / "baseline" / family / defn
    d = json.loads(next(sol_dir.glob("*.json")).read_text())
    spec = d.get("spec", {}); lang_s = str(spec.get("language", "python")).lower()
    lang_map = {"python": SupportedLanguages.PYTHON, "triton": SupportedLanguages.TRITON,
                "cuda": SupportedLanguages.CUDA, "cpp": SupportedLanguages.CPP}
    sol = Solution(
        name=d["name"], definition=defn, author=d.get("author", "baseline"),
        spec=BuildSpec(
            language=lang_map.get(lang_s, SupportedLanguages.PYTHON),
            target_hardware=list(spec.get("target_hardware") or []),
            entry_point=spec.get("entry_point", ""),
            dependencies=list(spec.get("dependencies") or []),
            binding=str(spec.get("binding", "torch")),
            destination_passing_style=bool(spec.get("destination_passing_style", True)),
        ),
        sources=[SourceFile(path=sf["path"], content=sf["content"]) for sf in d.get("sources", [])],
        description=d.get("description"),
    )
    print(f"\n=== {key} baseline={sol.name} ===")
    for s in info["selected"]:
        task = FlashInferBenchTask.from_cli_args(
            task_path=str(dataset_root), definition_name=defn,
            warmup_runs=2, iterations=5, num_trials=1, rtol=1e-2, atol=1e-2,
            use_isolated_runner=False, baseline_solution=None,
            feedback_workloads=[s["uuid"]], feedback_trace_policy="first",
            num_feedback_workloads=1, artifacts_dir=None, enable_ncu_profile=False,
        )
        er = task.run_benchmark(solution=sol, dump_traces=False, round_num=0)
        ok = str(er.status or "").lower() == "passed"
        print(f"  {'PASS' if ok else 'FAIL'}  {s['uuid']}  axes={s['axes']}  latency={er.latency_ms}")
PY
```

All 11 selected workloads (5 dsa_sparse + 6 dsa_topk) should report `PASS`.

---

## 3. LLM credentials

Two backends are supported. Pick **one** per run.

### 3.1 OpenAI-compatible HTTP API (default)

Put credentials in `.env` at the repo root (gitignored):

```
LLM_API_KEY=<your api key>
LLM_BASE_URL=<your provider's base URL, e.g. https://api.openai.com/v1>
```

`generate_kernels_and_eval.py` auto-loads `.env` at startup. Resolution order for both `api_key` and `base_url`:

1. CLI flag (`--api-key` / `--base-url`)
2. Env var (`LLM_API_KEY` / `LLM_BASE_URL`)
3. Env var (`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`)
4. Hard default: `https://api.anthropic.com/v1/` for base_url; api_key has no default (errors if missing).

### 3.2 Claude Code subscription (alternative)

If you have Claude Code installed and logged in (`claude login`), you can route every LLM call through `claude -p` subprocess by setting `USE_CLAUDE_CLI=1` when launching a script. No api key needed; uses your subscription. Note: concurrent K-Search processes share the same subscription throughput, so running two in parallel does not double throughput.

---

## 4. Running a search

5 ready-made scripts under `scripts/`, one per kernel:

| Script | Kernel | Baseline | Workloads |
|---|---|---|---|
| `scripts/dsa_sparse_attention.sh` | DSA sparse attention | `flashinfer_wrapper_5af199` | 5 |
| `scripts/dsa_topk_indexer.sh` | DSA topk indexer | `flashinfer_deepgemm_wrapper_2ba145` | 6 |
| `scripts/gdn_decode.sh` | GDN decode | `flashinfer_wrapper_9b7f1e` | 6 |
| `scripts/gdn_prefill.sh` | GDN prefill | `flashinfer_wrapper_123ca6` | 6 |
| `scripts/moe_fp8.sh` | MoE FP8 block-scale | `flashinfer_wrapper_9sdjf3` | 6 |

Curated workload UUIDs live in `scripts/selected_workloads.json` (single source of truth) and are baked into each script as `--feedback-workloads`.

### 4.1 Minimal launch

```bash
# OpenAI-compatible HTTP (api key from .env):
bash scripts/dsa_sparse_attention.sh

# Or Claude Code subscription:
USE_CLAUDE_CLI=1 bash scripts/dsa_sparse_attention.sh
```

### 4.2 Common env-var overrides

| Env var | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `claude-opus-4-6` | LLM model id |
| `API_KEY` | (from `.env`) | overrides `.env` |
| `BASE_URL` | (from `.env`) | overrides `.env` |
| `USE_CLAUDE_CLI` | unset (off) | when set, routes via `claude -p` |
| `LANGUAGE` | `cuda` | target language (`triton` / `cuda` / `python`) |
| `TARGET_GPU` | `H100` | prompt hint only |
| `MAX_OPT_ROUNDS` | `120` | sized for ~24h with `--use-claude-cli` + medium effort (~12 min/round) |
| `WM_STAGNATION_WINDOW` | `5` | rounds without improvement → end cycle, refine WM, persist checkpoint |
| `ARTIFACTS_DIR` | `.ksearch-output` | per-task subdir is created automatically — give each concurrent run its own dir |
| `WANDB_PROJECT` | unset (no project flag) | W&B project name |
| `WANDB_ENTITY` | unset (no entity flag) | W&B entity / team |
| `RUN_NAME` | `${MODEL}-${LANG}-wm-${DEFINITION}-seed-opt${ROUNDS}` | W&B run name |
| `CONTINUE_FROM_SOLUTION` | unset | resume from a previously-generated solution by name |

Example: longer run with W&B, separate artifacts dir:

```bash
WANDB_PROJECT=my_proj WANDB_ENTITY=my_team \
ARTIFACTS_DIR=.ksearch-output/dsa_topk \
MAX_OPT_ROUNDS=200 \
bash scripts/dsa_topk_indexer.sh
```

### 4.3 Auto-resume

All scripts pass `--auto-resume`. On every restart, K-Search loads:

- `<ARTIFACTS_DIR>/<definition>/world_model/world_model.json` — search tree
- `<ARTIFACTS_DIR>/<definition>/world_model/best_checkpoint.json` — best solution so far
- `<ARTIFACTS_DIR>/<definition>/world_model/progress.json` — round counter
- `<ARTIFACTS_DIR>/<definition>/wandb_run_id.txt` — W&B run id (continues same run with `resume="allow"`)

Per-cycle persistence: world_model.json is rewritten at every WM-tree mutation (cycle-end attach + refine, or note-too-hard). Best-checkpoint + progress are rewritten on every new best round. Crashes mid-cycle lose the in-progress attempts but the tree stays consistent.

---

## 5. Enterprise / batch deployment

Everything that affects a run is env-var-driven. A typical job spec:

```bash
# 1. setup once at image build time:
#    - uv sync
#    - clone /tmp/flashinfer-bench-main and /tmp/DeepGEMM (recursive)
#    - uv pip install -e /tmp/DeepGEMM --no-build-isolation
#    - symlink cutlass headers (see 1.3)
#    - download dataset to ~/dataset/flashinfer-trace
#    - apply DSA topk baseline patch (see 2.1)

# 2. per-job env injection:
export LLM_API_KEY=...                   # or USE_CLAUDE_CLI=1
export LLM_BASE_URL=...
export WANDB_API_KEY=...
export WANDB_PROJECT=...
export WANDB_ENTITY=...
export ARTIFACTS_DIR=/persistent/run-XYZ  # mount durable storage here so --auto-resume can pick up after preemption
export MAX_OPT_ROUNDS=...

# 3. launch one kernel per job (no concurrency benefit from claude-cli; with HTTP backend, run multiple jobs):
bash scripts/<kernel>.sh
```

Notes:
- **One kernel per job** is simplest. Each kernel has its own `ARTIFACTS_DIR` subdir; preempted jobs resume cleanly.
- **Durable storage**: mount `ARTIFACTS_DIR` on persistent volume so `--auto-resume` survives node failures.
- **Heartbeat**: each cycle end writes `world_model.json` and `progress.json`. Jobs that die mid-cycle resume from the start of that cycle (lose ≤ `WM_STAGNATION_WINDOW` attempts).
- **Time budget**: with `--use-claude-cli --effort medium` on Opus 4.7, expect ~12 min / round end-to-end (codegen ~9.5 min + eval ~10 s + amortized refine). `MAX_OPT_ROUNDS=120` fills ~24 h. With a faster HTTP backend, dial up `MAX_OPT_ROUNDS` accordingly.
