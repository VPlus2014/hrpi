from __future__ import annotations
# import torch
import numpy as np
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Sequence, TypeVar

_T_NDArr = TypeVar("_T_NDArr", np.ndarray, torch.Tensor)
_T = TypeVar("_T")


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


def space2box(space: spaces.Space, dtype=np.float32) -> spaces.Box:
    def _get_value_bound(space: spaces.Space):
        if isinstance(space, spaces.Discrete):
            return (
                np.array([space.start], dtype=dtype),
                np.array([space.start + space.n - 1], dtype=dtype),
            )
        elif isinstance(space, spaces.Box):
            return (space.low.astype(dtype), space.high.astype(dtype))
        elif isinstance(space, spaces.Tuple):
            low_list, high_list = [], []
            for i in range(len(space)):
                _min_ndarray, high_ndarray = _get_value_bound(space[i])
                low_list.append(_min_ndarray)
                high_list.append(high_ndarray)
            return (
                np.concatenate(low_list, axis=-1).astype(dtype),
                np.concatenate(high_list, axis=-1).astype(dtype),
            )
        elif isinstance(space, spaces.Dict):
            low_list, high_list = [], []
            for key in space.keys():
                _min_ndarray, high_ndarray = _get_value_bound(space[key])
                low_list.append(_min_ndarray)
                high_list.append(high_ndarray)
            return (
                np.concatenate(low_list, axis=-1).astype(dtype),
                np.concatenate(high_list, axis=-1).astype(dtype),
            )
        else:
            raise NotImplementedError

    low, high = _get_value_bound(space)
    return spaces.Box(low, high)


def flatten(
    space: spaces.Space, data: np.number | torch.Tensor | Sequence | OrderedDict
) -> torch.Tensor:
    if isinstance(data, np.number) and isinstance(space, spaces.Discrete):
        value_flattened = torch.asarray([data])
    elif isinstance(data, torch.Tensor) and isinstance(space, spaces.Box):
        value_flattened = data.reshape((-1,) + space.shape)
    elif isinstance(data, Sequence) and isinstance(space, spaces.Tuple):
        tensor_list: list[torch.Tensor] = []
        assert len(data) == len(space)
        for _value, _space in zip(data, space, strict=True):
            _value_flattened = flatten(_space, _value)
            tensor_list.append(_value_flattened)
        value_flattened = torch.cat(tensor_list, axis=-1)
    elif isinstance(data, OrderedDict) and isinstance(space, spaces.Dict):
        tensor_list: list[torch.Tensor] = []
        assert list(data.keys()) == list(space.keys())
        for _value, space_value in zip(data.values(), space.values(), strict=True):
            _value_flattened = flatten(space_value, _value)
            tensor_list.append(_value_flattened)

        value_flattened = torch.cat(tensor_list, axis=-1)
    else:
        raise NotImplementedError(
            "type(value) is {}, type(space) is {}".format(type(data), type(space))
        )

    return value_flattened


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


def discretize_space(space: spaces.Box, nvec: Sequence[int]):
    """
    离散化空间
    """

    if isinstance(space, spaces.Box):
        assert all(n > 0 for n in nvec), "nvec must be positive"
        return spaces.MultiDiscrete(np.array(nvec) - 1)
    else:
        raise TypeError("space must be Box, got", type(space))


def normalize(data: _T, low: _T, high: _T) -> _T:
    """
    线性归一化 [low, high]->[0, 1]
    Args:
        data (np.ndarray | torch.Tensor): _description_
        low (np.ndarray | torch.Tensor): 端点1
        high (np.ndarray | torch.Tensor): 端点2
    Returns:
        (np.ndarray | torch.Tensor): 归一化数据
    """
    data = (data - low) / (high - low)  # type: ignore
    return data


def affcmb(weight: _T, low: _T, high: _T) -> _T:
    """
    仿射组合(线性归一化的逆映射)
    $$
    low + w * (high - low)
    $$
    Args:
        weight (np.ndarray | torch.Tensor): 权重
        low (np.ndarray | torch.Tensor): 端点1
        high (np.ndarray | torch.Tensor): 端点2
    Returns:
        (np.ndarray | torch.Tensor): 仿射组合数据
    """
    return low + weight * (high - low)  # type: ignore
