from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path


def _setup():
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT = _setup()


import time
from typing import Any, cast
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from contextlib import ContextDecorator

from tools import init_seed, as_np, as_tsr, ConextTimer

from environments.models.aircraft.pdof6plane import PDOF6Plane as Plane
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def demo1(
    batch_size=1,
    max_steps=10000,
    use_tqdm=True,
    seed=0,
    device="cpu",
    dtype=torch.float32,
):
    """Sync Simulation"""
    rng = np.random.default_rng(seed)

    model = Plane(
        tas=240,
        rpy_ew=torch.zeros((batch_size, 3), device=device, dtype=dtype),
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    model.reset()

    qbar = range(max_steps)
    if use_tqdm:
        qbar = tqdm(qbar)
    for itr in qbar:
        action = rng.uniform(-1, 1, size=(batch_size, 4))
        model.set_action(action)


def demo2(
    pexc: ProcessPoolExecutor | ThreadPoolExecutor,
    device="cpu",
    dtype=torch.float32,
    batch_size=1,
    max_steps=10000,
    use_tqdm=True,
    seed=0,
):
    """Async Simulation"""
    tasks = [
        pexc.submit(
            demo1,
            batch_size=1,
            max_steps=max_steps,
            use_tqdm=False,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        for _ in range(batch_size)
    ]
    # 等待任务完成
    _k = 0
    _k0 = 0
    _t0 = time.time()
    dt_check = 1e-3
    _ndone0 = 0
    qbar = range(batch_size)
    if use_tqdm:
        qbar = tqdm(qbar)
    while True:
        _k = int((time.time() - _t0) // dt_check)
        if _k <= _k0:
            continue
        _k0 = _k
        _tsks_done = [tsk for tsk in tasks if tsk.done()]
        _ndone = len(_tsks_done)
        if _ndone == _ndone0:
            continue
        if isinstance(qbar, tqdm):
            qbar.update(_ndone - _ndone0)
        _ndone0 = _ndone
        if _ndone == batch_size:
            break


def main():
    batch_size = 1000
    config: dict[str, Any] = dict(
        device="cuda",
        dtype=torch.float32,
        batch_size=batch_size,
        max_steps=10,
        use_tqdm=True,
        seed=0,
    )

    nworkers = 4
    print(f"Using {nworkers} workers")

    for mode in ["sync", "async_process", "async_thread"]:
        t0 = time.time()
        if mode == "sync":
            demo1(**config)
        elif mode == "async_process":
            ppe = ProcessPoolExecutor(max_workers=nworkers)
            demo2(pexc=ppe, **config)
            ppe.shutdown()
        elif mode == "async_thread":
            tpe = ThreadPoolExecutor(max_workers=nworkers)
            demo2(pexc=tpe, **config)
            tpe.shutdown()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        dt = time.time() - t0
        fps = batch_size / max(dt, 1e-6)
        print(f"Mode: {mode}, Time elapsed: {dt:.2f}s, FPS: {fps:.2f}")
    return


if __name__ == "__main__":
    main()
