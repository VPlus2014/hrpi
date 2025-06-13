from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path


def _setup():  # 确保项目根节点在 sys.path 中
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

from environments_th.simulators.aircraft.p6dof import P6DOFPlane as Plane_th
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def run_kern(
    batch_size=1,
    max_steps=10000,
    use_tqdm=True,
    seed=0,
    device="cpu",
    dtype=torch.float32,
):
    """Sync Simulation"""
    rng = np.random.default_rng(seed)

    model = Plane_th(
        device=device,
        dtype=dtype,
        group_shape=(batch_size,),
    )
    model.set_ic_tas(240, None)
    model.set_ic_rpy_ew(0, None)
    model.reset(None)

    if use_tqdm:
        qbar = tqdm(range(max_steps * batch_size))
    for itr in range(max_steps):
        # action = rng.uniform(-1, 1, size=(batch_size, 4))
        # model.set_action(action)
        model.run(None)
        if use_tqdm:
            qbar.update(batch_size)
    return True


def run_mp(
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
            run_kern,
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
    if use_tqdm:
        qbar = tqdm(range(batch_size * max_steps))
    done = True
    try:
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
                qbar.update((_ndone - _ndone0) * max_steps)
            _ndone0 = _ndone
            if _ndone == batch_size:
                break
    except KeyboardInterrupt:
        done = True
        for tsk in tasks:
            tsk.cancel()
    return done


def main():
    """
    效率:
    Vec >> Subproc >> Thread, 约 batch_size:1, 10:1
    小规模下 CPU >> CUDA, 约 4:1, 仿真单位数 12800 附近则开始持平, 更大规模下 CUDA 更快
    """
    batch_size = 50
    max_steps = 10
    config: dict[str, Any] = dict(
        # device="cpu",
        dtype=torch.float32,
        batch_size=batch_size,
        max_steps=max_steps,
        use_tqdm=True,
        seed=0,
    )

    n_workers = 8
    print(f"Using {n_workers} workers")

    def _mode1(device):
        return run_kern(device=device, **config)

    def _mode2(device):
        pe = ProcessPoolExecutor(max_workers=n_workers)
        done = run_mp(pexc=pe, device=device, **config)
        pe.shutdown(True)
        return done

    def _mode3(device):
        pe = ThreadPoolExecutor(max_workers=n_workers)
        done = run_mp(pexc=pe, device=device, **config)
        pe.shutdown(True)
        return done

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    for mode, _run in [
        ("sync", _mode1),
        ("async_process", _mode2),
        # ("async_thread", _mode3),
    ]:
        for device in devices:
            try:
                t0 = time.time()
                done = _run(device=device)
                dt = time.time() - t0
                if done:
                    fps = (batch_size * max_steps) / max(dt, 1e-6)
                    print(
                        f"mode={mode},device={device}, Time elapsed: {dt:.3f}s, FPS: {fps:.2f}"
                    )
            except KeyboardInterrupt:
                pass
            except Exception as e:
                raise e
    return


if __name__ == "__main__":
    main()
