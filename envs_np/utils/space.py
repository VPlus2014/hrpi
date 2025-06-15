from __future__ import annotations

# import torch
import numpy as np
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Sequence, TypeVar
from .math_np import affcmb, affcmb_inv

_T_NDArr = TypeVar("_T_NDArr", np.ndarray, np.ndarray)
_T = TypeVar("_T", np.floating, np.integer, np.ndarray, int, float, bool)


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
    space 必须与 data 有相同的外层结构
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


def unflatten(space: spaces.Space, data: _T_NDArr):
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


def discretize_space(space: spaces.Box, nvec: Sequence[int] | np.ndarray):
    """
    离散化空间
    """

    if isinstance(space, spaces.Box):
        nvec = np.asarray(nvec)
        assert (nvec > 0).all(), ("nvec must be all positive", nvec)
        vld = np.isfinite(space.low)
        assert vld.all(), ("space.low must be all finite, but", np.where(~vld))
        vld = np.isfinite(space.high)
        assert vld.all(), ("space.high must be all finite, but", np.where(~vld))
        return spaces.MultiDiscrete(nvec)
    else:
        raise TypeError("space must be Box, got", type(space))


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
