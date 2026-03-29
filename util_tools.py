from __future__ import annotations
from contextlib import ContextDecorator
import time
from typing import Sequence, TypeVar, Union
import numpy as np
import torch
import random
import numpy
import os
from decimal import Decimal, getcontext

_T = TypeVar("_T")
TArr = Union[np.ndarray, torch.Tensor]
T_NDArr_co = TypeVar("T_NDArr_co", bound=TArr, covariant=True)

getcontext().prec = 4


def as_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)


def as_tsr(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.asarray(x)


def set_use_cuda_dsa(value: bool):
    """use device-side asserts"""
    os.environ["TORCH_USE_CUDA_DSA"] = str(int(value))


def set_num_threads(n: int):
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)


def set_cuda_launch_blocking(value: bool):
    """强制CUDA关闭异步"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(int(value))


class ConextTimer(ContextDecorator):
    def __init__(self, name: str = ""):
        self.name = name
        self.t = 0
        self.dt = 0
        self._lv = 0

    def reset(self):
        self.t = 0
        self.dt = 0

    def __enter__(self):
        self.push()

    def __exit__(self, *exc):
        self.pop()

    def push(self):
        self._lv += 1
        if self._lv == 1:
            self._t0 = time.time()

    def pop(self):
        if self._lv > 0:
            self._lv -= 1
            if self._lv == 0:
                self.dt = dt = time.time() - self._t0
                self.t += dt


def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    print(f"Seed initialized to {seed}")


def set_max_threads(n: int = 16):
    os.environ["NUMEXPR_MAX_THREADS"] = str(n)


def shape_rjust(x: T_NDArr_co, target: TArr) -> T_NDArr_co:
    """padding left with 1 to match target shape"""
    if x.ndim < target.ndim:
        shp = (1,) * (int(target.ndim) - int(x.ndim)) + x.shape
        return x.reshape(shp)  # type: ignore
    else:
        return x


def calc_range1d(pts: Sequence[_T] | np.ndarray) -> tuple[float, float]:
    if isinstance(pts, np.ndarray):
        vs = pts.ravel()  # type: ignore
    else:
        vs = np.hstack([np.ravel(p) for p in pts])  # type: ignore
    return np.min(vs), np.max(vs)


def fit_lim1d(vals: Sequence[_T] | np.ndarray, rr=0.05, rofst=1e-3):
    a, b = calc_range1d(vals)
    c = (b + a) * 0.5
    r = (b - a) * 0.5
    r = (r + rofst) * (1 + rr)
    assert r > 0, "range is too small"
    return c - r, c + r
