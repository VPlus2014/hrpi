from __future__ import annotations
from typing import Any
import numpy as np


# 螺线型轨迹生成器
def generate_cylindrical_spiral(
    r: np.ndarray | float | Any,
    kz: np.ndarray | float | Any,
    omega: np.ndarray | float | Any,
    t0: np.ndarray | float | Any,
    tf: np.ndarray | float | Any,
    n: int,
):
    """
    Generate a cylindrical spiral trajectory.
    Args:
        r: 半径, float|shape=(B, 1)
        k: z, float|shape=(B, 1)
        t0: 起始弧度, float|shape=(B, 1)
        tf: 终止弧度, float|shape=(B, 1)
        n: 轨迹点数, 必须大于等于2
    """
    assert n >= 2, "轨迹点数必须大于等于2"
    r = np.reshape(r, (-1, 1))  # (B, 1)
    kz = np.reshape(kz, (-1, 1))  # (B, 1)
    t0 = np.reshape(t0, (-1, 1))  # (B, 1)
    tf = np.reshape(tf, (-1, 1))  # (B, 1)
    omega = np.reshape(omega, (-1, 1))  # (B, 1)
    t = np.linspace(t0, tf, n, axis=-2).squeeze(-1)  # (B,n)

    _wt = omega * t  # (..., n)
    x = r * np.cos(_wt)  # (..., n)
    y = r * np.sin(_wt)  # (..., n)
    z = kz * t  # (..., n)
    x, y, z = np.broadcast_arrays(x, y, z)  # (..., n)
    xyz = np.stack([x, y, z], axis=-1)  # (..., n, 3)
    return xyz
