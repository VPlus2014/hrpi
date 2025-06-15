from __future__ import annotations
import time
from typing import Any

from tqdm import tqdm


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT = _setup()

from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import torch
from environments_th.simulators.aircraft.p6dof import P6DOFPlane as P6DOFPlane_th
from envs_np.simulators.aircraft.p6dof import P6DOFPlane as P6DOFPlane_np


def test(
    use_np=True,
    unit_num=1,
    device="cpu",
    use_float64=True,
    max_episodes=100,
    max_steps=100,
    seed=0,
    use_tqdm=True,
    title="Test",
):
    rng = np.random.default_rng(seed)
    if use_np:
        grp = P6DOFPlane_np(
            group_shape=(unit_num,),
            device=device,
            use_float64=use_float64,
        )
    else:
        grp = P6DOFPlane_th(
            group_shape=(unit_num,),
            device=device,
            dtype=torch.float64 if use_float64 else torch.float32,
        )
    qbar = range(max_episodes * max_steps)
    if use_tqdm:
        qbar = tqdm(qbar)
        qbar.set_description(title)
    for i in qbar:
        if i % max_steps == 0:
            grp.reset(None)

        grp.set_action(rng.random((unit_num, 4)))
        grp.run(None)

    del grp


def main():
    """
    FPS实验:
    unit_num    64      128*1   128*32  128*64  128*128 128*1024
    Torch@CUDA  4e3     7e3     29e4    55e4    100e4*  590e4*
    Torch@CPU   21e3    29e3    21e4    22e4    36e4    87e4
    Numpy       44e3*   39e3*   42e4*   56e4*   49e4    36e4

    结论:
    中小规模用 Numpy, 大规模用 Torch@CUDA
    """
    unit_num = 128 * 64
    # group_size = 1
    max_episodes = 10
    max_steps = 100
    config: dict[str, Any] = dict(
        unit_num=unit_num,
        # device="cuda",
        use_float64=True,
        use_tqdm=True,
        max_episodes=max_episodes,
        max_steps=max_steps,
        seed=int(time.time()),
    )
    total_steps = unit_num * max_steps * max_episodes

    for use_np, device in [
        (False, "cuda"),
        (False, "cpu"),
        (True, "cpu"),
    ]:
        t0 = time.time()
        test(
            use_np=use_np,
            device=device,
            title="{}@{}".format("Numpy" if use_np else "Torch", device),
            **config,
        )
        dt = time.time() - t0
        mode = {"use_numpy": use_np, "device": device}
        fps = total_steps / max(dt, 1e-6)
        print(f"Mode: {mode}, Time elapsed: {dt:.2f}s, Steps per second: {fps:.03g}\n")
    return


if __name__ == "__main__":
    main()
