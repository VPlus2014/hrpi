from __future__ import annotations
from typing import Sequence, SupportsInt
import numpy as np


def generate_waypoints(
    group_shape: Sequence[int],
    num: int,
    npad: int,
    nvec: Sequence[int] | np.ndarray,
    seed: int | None = None,
    dtype: type[np.floating | np.integer] = np.float64,
    use_random: bool = True,
    disc_table: Sequence[np.ndarray] | None = None,
    disc_max: int = 0,
) -> np.ndarray:
    r"""产生一组路径点
    TODO: 增加连续性约束，例如相邻点距、曲率，在约束下快速生成
    Args:
        group_shape:
        num: 有效路径点数
        npad: 额外填充路径点数(用最后一个有效路径点填充)
        nvec: 各维度的离散化位置数, shape=(d,)
        use_random: 是否随机生成路径点, 否则只生成原点
        disc_table: 每个维度的离散化位置表, disc_table[i].shape=(nvec[i],); None则不映射
        disc_max: 像素 $l_\infty$ 距离连续性约束, <=0 表示无约束
        ...
    Returns:
        waypoints: shape=group_shape+(num+npad, d).
    """
    grp_shape = (*group_shape,)
    grp_ndim = len(grp_shape)
    assert num > 0, ("num should be positive", num)
    assert npad >= 0, ("npad should be non-negative", npad)
    nvec = np.ravel(nvec).astype(np.intp)
    _d = len(nvec)
    if not use_random:
        goals = np.zeros(grp_shape + (num, _d), dtype=dtype)  # 只有原点
    else:
        _1s = (1,) * grp_ndim
        nvec = nvec.reshape(_1s + (1, -1))  # (...,1,d)
        rng = np.random.default_rng(seed)

        # TODO: 去重
        if disc_max > 0:
            dis = rng.integers(-disc_max, disc_max + 1, grp_shape + (num, _d))
            idxs = np.cumsum(dis, axis=-2)
            idxs = np.clip(idxs, 0, nvec - 1)
        else:
            idxs = rng.integers(0, nvec, grp_shape + (num, _d))  # (...,num,d)

        if disc_table is not None:
            assert len(disc_table) == _d, (
                "pos_disc_table should have length dim_pos",
                len(disc_table),
                _d,
            )
            disc_table = [
                np.reshape(tb, _1s + (1, -1)) for tb in disc_table
            ]  # [i]->(...,1,N_i)
            goals = [
                np.take_along_axis(
                    disc_table[i], idxs[..., i : i + 1], axis=-1
                )  # gather
                for i in range(_d)
            ]
            goals = np.concatenate(goals, axis=-1, dtype=dtype)  # (...,num,d)
        else:
            goals = idxs.astype(dtype)  # (...,num,d)

    # 最后一个导航点填充
    goals = np.concatenate(
        [goals, np.empty(grp_shape + (npad, _d), dtype=dtype)], axis=-2
    )
    goals[..., -npad + 1 :, :] = goals[..., [-npad], :]
    return goals
