from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

_NVTX_RANGE_NAME = "flashinfer_bench_ncu_profile"


def run_ncu(
    solution: Union[object, str],
    workload: Union[object, str],
    *,
    # Runtime
    device: str = "cuda:0",
    trace_set_path: Optional[str] = None,
    # NCU config, TODO: wrap it as a skill, let agent decide which part to be profiled
    set: str = "empty",
    sections: Optional[List[str]] = ["SpeedOfLight", "LaunchStats", "Occupancy"],
    kernel_name: Optional[str] = None,
    page: str = "details",
    ncu_path: str = "ncu",
    # Execution
    timeout: int = 600,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Run NCU profiling on a flashinfer-bench solution.

    Parameters
    ----------
    solution:
        A flashinfer_bench.data.Solution object or path to a JSON file.
    workload:
        A flashinfer_bench.data.Workload object or path to a JSON file.
    device:
        CUDA device string, e.g. ``"cuda:0"``.
    trace_set_path:
        Path to the flashinfer-bench trace-set.  Falls back to the
        ``FIB_DATASET_PATH`` environment variable when omitted.
    set:
        NCU section set (e.g. ``"detailed"``, ``"full"``).
    sections:
        Additional NCU sections beyond *set*.
    kernel_name:
        Optional regex to restrict profiling to matching kernel names.
    page:
        NCU output page: ``"raw"``, ``"details"``, or ``"source"``.
    ncu_path:
        Path to the ``ncu`` executable.
    timeout:
        Subprocess timeout in seconds.
    tmpdir:
        Directory for temporary build artefacts.
    max_lines:
        Truncate output to at most this many lines (``None`` = unlimited).

    Returns
    -------
    str
        Raw NCU text output, or a string starting with ``"ERROR:"`` on
        failure.
    """
    from flashinfer_bench.data import Solution, TraceSet, Workload

    # --- resolve solution ---
    if isinstance(solution, str):
        p = Path(solution)
        if not p.exists():
            return f"ERROR: Solution file not found: {solution}"
        try:
            solution = Solution.model_validate_json(p.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse solution: {e}"

    # --- resolve workload ---
    if isinstance(workload, str):
        p = Path(workload)
        if not p.exists():
            return f"ERROR: Workload file not found: {workload}"
        try:
            workload = Workload.model_validate_json(p.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse workload: {e}"

    # --- validate page ---
    if page not in {"raw", "details", "source"}:
        return f"ERROR: Invalid page '{page}'. Must be one of: raw, details, source"

    # --- load trace set ---
    try:
        trace_set = TraceSet.from_path(trace_set_path)
    except Exception as e:
        return f"ERROR: Failed to load trace set: {e}"

    if solution.definition not in trace_set.definitions:
        avail = list(trace_set.definitions.keys())
        return (
            f"ERROR: Definition '{solution.definition}' not in trace set. "
            f"Available: {avail}"
        )
    definition = trace_set.definitions[solution.definition]

    # --- check ncu binary ---
    if shutil.which(ncu_path) is None:
        return f"ERROR: NCU not found at '{ncu_path}'. Install NVIDIA Nsight Compute."

    with tempfile.TemporaryDirectory(prefix="ksearch_ncu_", dir=tmpdir) as build_dir:
        build_path = Path(build_dir)

        (build_path / "definition.json").write_text(definition.model_dump_json())
        (build_path / "solution.json").write_text(solution.model_dump_json())
        (build_path / "workload.json").write_text(workload.model_dump_json())

        cmd = _build_cmd(
            build_path=build_path,
            set=set,
            sections=sections,
            kernel_name=kernel_name,
            page=page,
            device=device,
            trace_set_path=trace_set_path,
            ncu_path=ncu_path,
        )

        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        logger.info("run_ncu command: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: NCU timed out after {timeout}s."

        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            return f"ERROR: NCU exited {proc.returncode}:\n{output}"

        if max_lines is not None:
            lines = output.split("\n")
            if len(lines) > max_lines:
                extra = len(lines) - max_lines
                output = "\n".join(lines[:max_lines])
                output += f"\n[truncated: {extra} more lines]"

        return output


def _build_cmd(
    *,
    build_path: Path,
    set: str,
    sections: Optional[List[str]],
    kernel_name: Optional[str],
    page: str,
    device: str,
    trace_set_path: Optional[str],
    ncu_path: str,
) -> List[str]:
    cmd = [
        ncu_path,
        "--page", page,
        "--set", set,
        "--nvtx",
        "--nvtx-include", _NVTX_RANGE_NAME,
        "-f",
    ]

    if sections:
        for s in sections:
            cmd.extend(["--section", s])

    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])

    # Use our own runner (start/end NVTX) instead of flashinfer_bench's runner.
    runner = [
        sys.executable, "-u", "-m", "k_search.utils._ncu_runner",
        "--data-dir", str(build_path),
        "--device", device,
    ]
    if trace_set_path:
        runner.extend(["--trace-set-path", str(trace_set_path)])

    cmd.extend(runner)
    return cmd
