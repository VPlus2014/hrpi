from __future__ import annotations
from functools import partial
import time
from timeit import timeit
from typing import Sequence, TypeVar, Union
import numpy as np
from numba import njit, jit
from numpy.typing import NDArray
import torch


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT = _setup()

from codes.envs_np.utils.space_tf import undisc_with_gather, undisc_with_interp

_TInt = TypeVar("_TInt", bound=np.integer | int, covariant=True)
_T_ArrOrNum = Union[Sequence[_TInt], _TInt]
_Int_NDArr = NDArray[np.integer]
_IndexType4Gather = Union[
    _T_ArrOrNum[int],
    _T_ArrOrNum[np.integer],
    _Int_NDArr,
]


# !@jit
# @njit
def undisc_gather(
    vd: _Int_NDArr,
    table: Sequence[NDArray],
    nvec: _Int_NDArr,
    low: NDArray,
    high: NDArray,
) -> NDArray:
    v0 = undisc_with_gather(vd, table)
    return v0


def undisc_affine(
    vd: _Int_NDArr,
    table: Sequence[NDArray],
    nvec: _Int_NDArr,
    low: NDArray,
    high: NDArray,
) -> NDArray:
    y = undisc_with_interp(vd, nvec, low, high)
    return y


def get_qa(qs: torch.Tensor, a: torch.Tensor):  # (...,dimV,sizeA)
    assert qs.shape[:-1] == a.shape[: qs.ndim - 1]

    if a.ndim < qs.ndim:
        a = a.clone()
    while a.ndim < qs.ndim:
        a = a.unsqueeze_(-1)
    a = a.broadcast_to(qs.shape[:-1] + (1,))  # (...,dimV,1)
    qa = qs.gather(-1, a)  # (...,dimV,1)
    # assert qa.shape[:-1] == qs.shape[:-1]
    return qa


def get_qa_fast(qs: torch.Tensor, a: torch.Tensor):  # (...,dimV,sizeA)
    assert qs.shape[:-1] == a.shape[: qs.ndim - 1]
    while a.ndim < qs.ndim:
        a = a.unsqueeze(-1)
    a = a.broadcast_to(qs.shape[:-1] + (1,))  # (...,dimV,1)
    qa = qs.gather(-1, a)  # (...,dimV,1)
    # assert qa.shape[:-1] == qs.shape[:-1]
    return qa


def main():
    """
    耗时: gather>>affine, 至少是 2:1, 主要受 len(nvec)=dimA 影响
    """
    from tqdm import tqdm

    T = 1000
    B = 256
    seed = int(time.time())
    nvec = np.asarray([100] * 40, dtype=np.intp)
    dimA = len(nvec)
    dtype = np.intp
    r = 100.0
    #

    #
    nt1 = 100
    nt2 = 10
    for dtype in [np.intp, np.float32, np.float64]:
        err_tol = 1 if np.issubdtype(dtype, np.integer) else 1e-3
        for _f in [undisc_gather, undisc_affine]:
            rng = np.random.default_rng(seed)
            t = 0
            fname = _f.__name__
            qbar = tqdm(range(nt1))
            qbar.set_description(f"{fname}")
            #
            #
            #
            for _ in qbar:
                low = (rng.uniform(size=(dimA,)) * r).astype(dtype)
                high = (low + rng.uniform(size=(dimA,)) * r).astype(dtype)
                table = [
                    np.linspace(a_, b_, _n, dtype=dtype)
                    for a_, b_, _n in zip(low, high, nvec, strict=True)
                ]
                ad = rng.integers(nvec, size=(1, dimA)).astype(np.intp)
                a01 = undisc_gather(ad, table, nvec, low, high)
                a02 = undisc_affine(ad, table, nvec, low, high)
                err = a01 - a02

                assert np.abs(err).max() <= err_tol, f"err={err}"
                #
                args = (ad, table, nvec, low, high)
                _fwrapped = partial(_f, *args)
                t += timeit(_fwrapped, number=nt2)
            trajps = B * nt1 * nt2 / t
            print(f"{dtype.__name__} {fname}: {t:.5g} s, {trajps:.2f} traj/s\n")


if __name__ == "__main__":
    main()
