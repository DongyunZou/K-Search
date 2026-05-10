"""Cross-process GPU mutex via fcntl.flock.

Serializes evaluator entry points (run_benchmark / run_final_evaluation / run_ncu_profile)
across multiple K-Search processes that share a GPU. LLM calls do not hold the lock,
so other searches can interleave their evaluations while one process is waiting on a
model response.

The lock auto-releases on process exit (kernel-managed), so a crash will not leave the
GPU permanently locked.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path
from typing import Iterator, Optional


@contextlib.contextmanager
def gpu_lock(path: Optional[str], *, label: str = "eval") -> Iterator[None]:
    """Acquire an exclusive flock on ``path`` for the duration of the with-block.

    No-op when ``path`` is falsy (lock disabled). Logs acquire/wait/release events so
    multi-process interleaving is visible in the run log.
    """
    if not path:
        yield
        return
    p = Path(str(path)).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    pid = os.getpid()
    held_t0: Optional[float] = None
    try:
        t0 = time.monotonic()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(f"[gpu_lock] acquired {label} pid={pid} lock={p}", flush=True)
        except BlockingIOError:
            print(f"[gpu_lock] waiting  {label} pid={pid} lock={p}", flush=True)
            fcntl.flock(fd, fcntl.LOCK_EX)
            wait_s = time.monotonic() - t0
            print(
                f"[gpu_lock] acquired {label} pid={pid} after {wait_s:.1f}s lock={p}",
                flush=True,
            )
        held_t0 = time.monotonic()
        yield
    finally:
        held_s = (time.monotonic() - held_t0) if held_t0 is not None else 0.0
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass
        print(
            f"[gpu_lock] released {label} pid={pid} held={held_s:.1f}s lock={p}",
            flush=True,
        )
