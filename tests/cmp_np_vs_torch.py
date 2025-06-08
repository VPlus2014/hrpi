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
from environments.utils import math_np, math_pt


class PDOF6Group:
    def __init__(
        self,
        env_size: int = 1,
        group_size: int = 1,
        device: str = "cpu",
        np_float=np.float32,
        tsr_float: torch.dtype = torch.float32,
        use_numpy: bool = True,
    ):
        self._device = _device = torch.device(device)
        _bkbn = np if use_numpy else torch
        self._use_numpy = use_numpy
        if use_numpy:
            _1f1 = np.ones([env_size, group_size, 1], np_float)
            _0f1 = np.zeros([env_size, group_size, 1], np_float)
            _0f3 = np.zeros([env_size, group_size, 3], np_float)
            _0f4 = np.zeros([env_size, group_size, 4], np_float)
            _e1 = np.concatenate([_1f1, _0f1, _0f1], axis=-1)
            _e2 = np.concatenate([_0f1, _1f1, _0f1], axis=-1)
            _e3 = np.concatenate([_0f1, _0f1, _1f1], axis=-1)
        else:
            _1f1 = torch.ones(
                [env_size, group_size, 1], device=_device, dtype=tsr_float
            )
            _0f1 = torch.zeros(
                [env_size, group_size, 1], device=_device, dtype=tsr_float
            )
            _0f3 = torch.zeros(
                [env_size, group_size, 3], device=_device, dtype=tsr_float
            )
            _0f4 = torch.zeros(
                [env_size, group_size, 4], device=_device, dtype=tsr_float
            )
            _e1 = torch.cat([_1f1, _0f1, _0f1], dim=-1)
            _e2 = torch.cat([_0f1, _1f1, _0f1], dim=-1)
            _e3 = torch.cat([_0f1, _0f1, _1f1], dim=-1)
        self._e1 = _e1
        self._e2 = _e2
        self._e3 = _e3
        self._1f1 = _1f1
        self._0f1 = _0f1

        self._pos_e = _0f3 + 0.0
        self._vel_e = _0f3 + 0.0
        self._tas = _0f1 + 0.0
        self._Qew = _0f4 + 0.0
        self._rpy = _0f3 + 0.0
        self._g = _0f3 + 9.8
        self._g_e = _e3 * self._g

        self._ic_pos = _0f3 + 0.0
        self._ic_tas = _0f1 + 300
        self._ic_rpy = _0f3 + 0.0

        self._t = _0f1 + 0.0

        self._n_w = _0f3 + 0.0
        self._dmu = _0f1 + 0.0
        self._bkbn = math_np if use_numpy else math_pt
        self.use_gravity = True

        self._Vmin = 100
        self._Vmax = 1000

    def reset(self):
        bkbn = self._bkbn
        self._pos_e[...] = self._ic_pos
        self._tas[...] = self._ic_tas
        self._rpy[...] = self._ic_rpy
        self._Qew[...] = bkbn.rpy2quat(self._rpy)
        self._t[...] = 0.0
        self._ppgt()

    def run(self):
        bkbn = math_np if self._use_numpy else math_pt
        h = 1e-3
        pos, tas, Qeb, roll = bkbn.ode_rk45(
            self._f,
            self._t,
            [self._pos_e, self._tas, self._Qew, self._rpy[..., [0]]],
            h,
        )
        self._t[...] += h
        self._pos_e[...] = pos
        self._Qew[...] = Qeb
        self._rpy[..., 0:1] = roll

        Qeb = bkbn.normalize(Qeb)
        tas = bkbn._clip(tas, self._Vmin, None)
        self._ppgt()

    def _f(self, t, X):
        bkbn = self._bkbn
        use_np = self._use_numpy
        p_e, tas, Qew, mu = X

        dmu = self._dmu
        n_w = self._n_w
        _0 = self._0f1

        a_w = self._g * n_w  # 过载加速度风轴分量
        if self.use_gravity:
            Qwe = bkbn.quat_conj(Qew)
            a_w += bkbn.quat_rotate(Qwe, self._g_e)

        # 旋转角速度
        tas = bkbn._clip(tas, self._Vmin, self._Vmax)  # 防止过零
        Vinv = 1 / tas
        dot_tas = a_w[..., 0:1]
        a_vy = a_w[..., 1:2]
        a_vz = a_w[..., 2:3]
        P = dmu
        Q = -a_vz * Vinv
        R = a_vy * Vinv
        if use_np:
            h_w = np.concatenate([_0, P, Q, R], axis=-1) * 0.5
            v_w = np.concatenate([tas, _0, _0], axis=-1)
        else:
            h_w = torch.cat([_0, P, Q, R], dim=-1) * 0.5
            v_w = torch.cat([tas, _0, _0], dim=-1)
        dot_Qew = bkbn.quat_mul(Qew, h_w)

        dot_p_e = bkbn.quat_rotate(Qew, v_w)  # 惯性速度

        dotX = [dot_p_e, dot_tas, dot_Qew, dmu]
        return dotX

    def _ppgt(self):
        bkbn = math_np if self._use_numpy else math_pt
        self._rpy[...] = bkbn.rpy2quat_inv(self._Qew, self._rpy[..., [0]])
        self._vel_e[...] = bkbn.quat_rotate(self._Qew, self._tas * self._e1)


def test(
    use_np=True,
    env_size=1,
    group_size=1,
    device="cpu",
    tsr_float=torch.float32,
    np_float=np.float32,
    max_episodes=100,
    max_steps=100,
    seed=0,
    use_tqdm=True,
    title="Test",
):
    grp = PDOF6Group(
        env_size=env_size,
        group_size=group_size,
        device=device,
        np_float=np_float,
        tsr_float=tsr_float,
        use_numpy=use_np,
    )
    qbar = range(max_episodes)
    if use_tqdm:
        qbar = tqdm(qbar)
        qbar.set_description(title)
    for ep in qbar:
        grp.reset()
        for step in range(max_steps):
            grp.run()


def main():
    # 结论: 效率 Torch@CUDA >> Numpy > Torch@CPU
    env_size = 256
    group_size = 100
    config: dict[str, Any] = dict(
        # device="cuda",
        tsr_float=torch.float64,
        np_float=np.float64,
        env_size=env_size,
        group_size=group_size,
        use_tqdm=True,
        seed=0,
    )
    batch_size = env_size * group_size

    for use_np, device in [
        (False, "cuda"),
        (False, "cpu"),
        (True, "cpu"),
    ]:
        t0 = time.time()
        test(
            use_np=use_np,
            device=device,
            max_episodes=10,
            max_steps=100,
            title="{}@{}".format("Numpy" if use_np else "Torch", device),
            **config,
        )
        dt = time.time() - t0
        mode = {"use_numpy": use_np, "device": device}
        fps = batch_size / max(dt, 1e-6)
        print(f"Mode: {mode}, Time elapsed: {dt:.2f}s, FPS: {fps:.2f}")
    return


if __name__ == "__main__":
    main()
