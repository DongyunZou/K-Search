"""Standalone runner script invoked by ncu.py under ncu.

Mirrors flashinfer_bench.agents._solution_runner but emits NVTX
start/end ranges (nvtxRangeStartA/nvtxRangeEnd) instead of push/pop
(torch.cuda.nvtx.range).  NCU's --nvtx-include filter only matches
start/end ranges; push/pop ranges are silently ignored, which is why
the upstream runner yields "No kernels were profiled".
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch

from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Workload

_NVTX_RANGE_NAME = b"flashinfer_bench_ncu_profile"


def _nvtx_start() -> tuple:
    """Open an NVTX *start/end* range and return (lib, handle)."""
    lib = ctypes.CDLL("libnvToolsExt.so.1", use_errno=True)
    handle = lib.nvtxRangeStartA(_NVTX_RANGE_NAME)
    return (lib, handle)


def _nvtx_end(state: tuple) -> None:
    lib, handle = state
    lib.nvtxRangeEnd(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solution for NCU profiling")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-set-path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trace_set_path = Path(args.trace_set_path) if args.trace_set_path else None

    definition = Definition.model_validate_json((data_dir / "definition.json").read_text())
    solution = Solution.model_validate_json((data_dir / "solution.json").read_text())
    workload = Workload.model_validate_json((data_dir / "workload.json").read_text())

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, trace_set_path)

    inputs = gen_inputs(definition, workload, args.device, safe_tensors)
    outputs = allocate_outputs(definition, inputs, args.device)

    # Warmup — not captured by NCU because it is outside the NVTX range.
    with torch.no_grad():
        runnable.call_destination_passing(*inputs, *outputs)
    torch.cuda.synchronize()

    # Profiled run — wrapped in an NVTX *start/end* range so NCU
    # --nvtx-include can match it.
    nvtx_state = _nvtx_start()
    try:
        with torch.no_grad():
            runnable.call_destination_passing(*inputs, *outputs)
        torch.cuda.synchronize()
    finally:
        _nvtx_end(nvtx_state)

    runnable.cleanup()


if __name__ == "__main__":
    main()
