from __future__ import annotations
from typing import SupportsIndex, Union, overload
import torch
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils.spaces import batch_space as _batch_space
from collections import OrderedDict
from typing import Any, Sequence, TypeVar
from numpy.typing import NDArray
from ...utils.math_np import affcmb_inv
from numba import njit

_T_NDArr = TypeVar("_T_NDArr", bound=np.ndarray)
_T = TypeVar("_T", np.floating, np.integer, np.ndarray, int, float, bool)
_DType_co = TypeVar("_DType_co", bound=np.generic, covariant=True)
_TNPNum = TypeVar("_TNPNum", bound=np.number, covariant=True)
_TNumDTypeLike_co = Union[type[_TNPNum], np.dtype[_TNPNum]]


@overload
def batch_space(space: spaces.Box | spaces.MultiDiscrete, /, n: int) -> spaces.Box: ...


@overload
def batch_space(space: spaces.Discrete, /, n: int) -> spaces.MultiDiscrete: ...


def batch_space(space: spaces.Space, /, n: int):
    r"""Create a (batched) space, containing multiple copies of a single space.
    see `gymnasium.vector.utils.spaces.batch_space`
    """
    return _batch_space(space, n=n)


def get_spaces_shape(space: spaces.Space) -> int:
    if isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape).item()
    elif isinstance(space, spaces.Tuple):
        shape_list = [get_spaces_shape(subspace) for subspace in space]
        return sum(shape_list)
    elif isinstance(space, spaces.Dict):
        shape_list = [get_spaces_shape(subspace) for subspace in space.values()]
        return sum(shape_list)
    raise NotImplementedError("type(space) {} is unsupported".format(type(space)))


def _get_value_bound(space: spaces.Space, dtype: type[np.floating] = np.float32):
    if isinstance(space, spaces.Discrete):
        return (
            np.array([space.start], dtype=dtype),
            np.array([space.start + space.n - 1], dtype=dtype),
        )
    elif isinstance(space, spaces.Box):
        return (
            space.low.astype(dtype).ravel(),  # (d,)
            space.high.astype(dtype).ravel(),
        )
    elif isinstance(space, spaces.Tuple):
        low_list, high_list = [], []
        for i in range(len(space)):
            low, high = _get_value_bound(space[i])
            low_list.append(low)
            high_list.append(high)
        return (
            np.concatenate(low_list, axis=-1).astype(dtype),
            np.concatenate(high_list, axis=-1).astype(dtype),
        )
    elif isinstance(space, spaces.Dict):
        low_list, high_list = [], []
        for key in space.keys():
            low, high = _get_value_bound(space[key])
            low_list.append(low)
            high_list.append(high)
        return (
            np.concatenate(low_list, axis=-1).astype(dtype),
            np.concatenate(high_list, axis=-1).astype(dtype),
        )
    else:
        raise NotImplementedError(
            "unsupported space type", type(space), "@", _get_value_bound.__name__
        )


def space2box(space: spaces.Space, dtype: type[np.floating] = np.float32) -> spaces.Box:
    low, high = _get_value_bound(space, dtype=dtype)  # (d,)
    return spaces.Box(low, high, dtype=dtype)


def flatten(
    space: spaces.Space,
    data: np.number | np.ndarray | Sequence | OrderedDict | dict,
) -> np.ndarray:
    """
    批量拉平
    assert: 非向量化的 space 必须与 data 有相同的外层结构, such as Sequence, Dict
    Args:
        space: 空间
        data: 数据组

    Returns:
        output: 拉平后的数组, shape=(batch_size, dim(flattened_space))
    """
    if isinstance(space, spaces.Discrete):
        vec = np.asarray(data).reshape((-1, 1))
    elif isinstance(space, spaces.MultiDiscrete):
        assert len(space.nvec) == 1, (
            "MultiDiscrete space must be 1D tensor, got",
            len(space.nvec),
        )
        vec = np.asarray(data).reshape((-1, len(space.nvec)))
    elif isinstance(space, spaces.Box):
        assert len(space.shape) == 1, (
            "Box space must be 1D tensor, got",
            len(space.shape),
        )
        vec = np.asarray(data).reshape((-1,) + space.shape)
    elif isinstance(space, spaces.Tuple):
        tensor_list: list[np.ndarray] = []
        assert isinstance(data, (list, tuple)), (
            "data must be list or tuple",
            type(data),
        )
        assert len(data) == len(space), (
            "lenght of data and space must be equal",
            len(data),
            len(space),
        )
        for _src, _space in zip(data, space, strict=True):
            _vec = flatten(_space, _src)
            tensor_list.append(_vec)
        vec = np.concatenate(tensor_list, axis=-1)
    elif isinstance(space, spaces.Dict):  # DictSpace 的字典是 OrderedDict
        assert isinstance(data, (dict, OrderedDict)), (
            "data must be dict or OrderedDict",
            type(data),
        )
        tensor_list: list[np.ndarray] = []
        for key, _space in space.items():
            _src = data[key]
            _vec = flatten(_space, _src)
            tensor_list.append(_vec)
        vec = np.concatenate(tensor_list, axis=-1)
    else:
        raise NotImplementedError(
            "type(value) is {}, type(space) is {}".format(type(data), type(space))
        )
    return vec


def unflatten(
    space: spaces.Space,
    data: NDArray[_DType_co],
) -> tuple[Any, NDArray[_DType_co]]:
    if isinstance(space, spaces.Discrete):
        return data[..., 0:1], data[..., 1:]

    elif isinstance(space, spaces.Box):
        size = space.shape[0]
        return data[..., :size], data[..., size:]

    elif isinstance(space, spaces.Tuple):
        y1: list[Any] = []
        for _space in space:
            _data_unflattended, data = unflatten(_space, data)
            y1.append(_data_unflattended)
        return tuple(y1), data

    elif isinstance(space, spaces.Dict):
        data_unflattened: OrderedDict[str, Any] = OrderedDict()
        for key, _space in space.items():
            data_unflattened[key], data = unflatten(_space, data)

        return data_unflattened, data
    else:
        raise NotImplementedError("type(space) {} is unsupported".format(type(space)))


def discretize_value(
    cont_space: spaces.Box,
    disc_space: spaces.Space,
    value: np.ndarray,
):
    """
    离散化值
    """
    t = affcmb_inv(cont_space.low, cont_space.high, value)  # -> [0,1]
    if isinstance(disc_space, spaces.MultiDiscrete):
        dmax = disc_space.nvec - 1
    elif isinstance(disc_space, spaces.MultiBinary):
        dmax = 1
    elif isinstance(disc_space, spaces.Discrete):
        dmax = disc_space.n - 1
    else:
        raise TypeError("unsupported disc_space type", type(disc_space))
    value_d = np.round(t * dmax).astype(np.intp)
    return value_d


def _linspace4disc(
    low: np.ndarray,
    high: np.ndarray,
    n: SupportsIndex,
    dtype: _TNumDTypeLike_co | None = None,
):
    vals = np.linspace(low, high, n)
    dtype = dtype or vals.dtype.type
    # fix
    if np.issubdtype(dtype, np.integer):
        vals = np.round(vals)  # floor'll be unfair in end points
    vals = np.clip(vals, low, high)
    vals = vals.astype(dtype)
    return vals


def discretize_space(
    space: spaces.MultiDiscrete | spaces.Box | spaces.Discrete | spaces.Space,
    nvec: Sequence[int] | np.ndarray,
    dtype: type[np.integer] = np.intp,
) -> tuple[spaces.MultiDiscrete, list[np.ndarray]]:
    """
    N-D Box/Discrete -> N-D MultiDiscrete with table
    """
    nvec = np.asarray(nvec)
    assert nvec.ndim == 1, ("nvec must be 1D tensor, got ndim=", nvec.ndim)
    assert (nvec > 1).all(), ("nvec must be all greater than 1", nvec)
    dimA = len(nvec)
    space_d = spaces.MultiDiscrete(np.array(nvec, dtype=dtype), dtype=dtype)
    if isinstance(space, spaces.Box):
        src_ndim = len(space.shape)
        assert src_ndim == 1, ("Box space must be 1-dimensional", src_ndim)
        dimA0 = space.shape[0]
        assert dimA == dimA0, ("expected len(nvec)==space.shape[0], got", dimA, dimA0)
        table = [
            _linspace4disc(space.low[i], space.high[i], nvec[i], dtype=space.dtype)
            for i in range(dimA)
        ]
    elif isinstance(space, spaces.MultiDiscrete):
        src_ndim = len(space.nvec.shape)
        assert src_ndim == 1, ("MultiDiscrete nvec must be 1D tensor", src_ndim)
        dimA_ = space.nvec.shape[0]
        assert dimA == dimA_, ("expected len(nvec)==len(space.nvec), got", dimA, dimA_)
        table = [
            np.linspace(0, space.nvec[i] - 1, nvec[i], dtype=space.dtype)
            for i in range(dimA)
        ]
    elif isinstance(space, spaces.Discrete):
        assert dimA == 1, ("expected len(nvec)==1, got", dimA)
        table = [
            np.linspace(
                space.start,
                space.start + space.n - 1,
                nvec[0],
            )
        ]

    else:
        raise TypeError("space must be Box, got", type(space))
    return space_d, table


# !@njit
def undisc_with_gather(
    value_d: np.ndarray,
    table: Sequence[np.ndarray],
):
    """
    反离散化值(查表式)
    Args:
        table: 各维离散化表
            assert: len(table)==d;
            assert: len(table[i])==nvec[i] forall i;
            assert: table[i].ndim==1 forall i;
        value_d: discretized value, shape=(..., d)
    Returns:
        y: undiscretized value, shape=(..., d)
    """
    dimX = len(table)
    assert dimX == value_d.shape[-1], (
        "expected len(table)==value_d.shape[-1], got",
        dimX,
        value_d.shape[-1],
    )
    _shp = (1,) * len(value_d.shape[:-1]) + (-1,)
    ys = [
        np.take_along_axis(table[i].reshape(_shp), value_d[..., i : i + 1], axis=-1)
        for i in range(dimX)
    ]
    y = np.concatenate(ys, axis=-1)
    return y


def undisc_with_interp(
    value_d: np.ndarray,
    nvec: Sequence[int] | np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    dtype: type[np.number | np.bool_] | None = None,
):
    """
    反离散化值(线性插值式)
    Args:
        value_d: discretized value, shape=(..., d)
        low: lower bound of each dimension, shape=(d,)
        high: upper bound of each dimension, shape=(d,)
    Returns:
        y: undiscretized value, shape=(..., d)
    """

    ndim = value_d.ndim
    nvec = np.ravel(nvec)
    # assert (nvec>1).all(), "nvec must be all greater than 1"
    nvec = nvec.reshape((1,) * (ndim - nvec.ndim) + (-1,))  # (..., dimA|1)

    low = np.ravel(low)
    high = np.ravel(high)
    low = low.reshape((1,) * (ndim - low.ndim) + (-1,))  # (..., dimA|1)
    high = high.reshape((1,) * (ndim - high.ndim) + (-1,))  # (..., dimA|1)
    span = high - low

    t = affcmb_inv(0, nvec - 1, value_d)
    y = t * span + low
    assert y.ndim == value_d.ndim, (
        "expected y.ndim==value_d.ndim, got",
        y.ndim,
        value_d.ndim,
    )
    dtype = dtype or span.dtype.type
    y = y.astype(dtype)
    y = np.clip(y, low, high)
    return y
