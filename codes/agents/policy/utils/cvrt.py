from __future__ import annotations
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Literal,
    Sequence,
    TypeVar,
    cast,
    overload,
)
from typing import Union, no_type_check
import numpy as np
from numpy.typing import NDArray
import torch
from tianshou.data.batch import (
    Batch,
    _is_batch_set,
    _is_number,
    _to_array_with_correct_type,
    BatchProtocol,
)

_Number = Union[Number, np.number, np.bool_, int, float, bool]
_NumNDArr = Union[NDArray[Union[np.number, np.bool_]], torch.Tensor]

_T_Num_co = TypeVar(
    "_T_Num_co",
    bound=_Number,
    covariant=True,
)
_T_NumNDArr = TypeVar(
    "_T_NumNDArr",
    bound=_NumNDArr,
    covariant=True,
)

_BatchInputTypes = TypeVar(
    "_BatchInputTypes",
    bound=Union[dict, BatchProtocol, Sequence[Union[dict, BatchProtocol]], np.ndarray],
    covariant=True,
)


def _parse_value(obj: Any) -> Batch | np.ndarray | torch.Tensor | None:
    if isinstance(obj, Batch):  # most often case
        return obj
    if (
        (
            isinstance(obj, np.ndarray)
            and np.issubdtype(obj.dtype, type[np.bool_ | np.number])
        )
        or isinstance(obj, torch.Tensor)
        or obj is None
    ):  # third often case
        return obj
    if _is_number(obj):  # second often case, but it is more time-consuming
        return np.asanyarray(obj)
    if isinstance(obj, dict):
        return Batch(obj)
    if (
        not isinstance(obj, np.ndarray)
        and isinstance(obj, Collection)
        and len(obj) > 0
        and all(isinstance(element, torch.Tensor) for element in obj)
    ):
        try:
            obj = cast(list[torch.Tensor], obj)
            return torch.stack(obj)
        except RuntimeError as exception:
            raise TypeError(
                "Batch does not support non-stackable iterable"
                " of torch.Tensor as unique value yet.",
            ) from exception
    if _is_batch_set(obj):  # list of dict / Batch
        y = Batch(obj)  # type: ignore
        return y
    else:
        # None, scalar, normal obj list (main case)
        # or an actual list of objects
        try:
            obj = _to_array_with_correct_type(obj)
        except ValueError as exception:
            raise TypeError(
                "Batch does not support heterogeneous list/tuple of tensors as unique value yet.",
            ) from exception
    return obj


@overload
def to_numpy(x: dict | Batch) -> Batch: ...


@overload
def to_numpy(
    x: _NumNDArr | _Number,
) -> np.ndarray: ...


# TODO: confusing name, could actually return a batch...
#  Overrides and generic types should be added
# todo check for ActBatchProtocol
@no_type_check
def to_numpy(x: Any) -> Batch | np.ndarray:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):  # second often case
        return x
    if isinstance(x, np.number | np.bool_ | Number):
        return np.asanyarray(x)
    if x is None:
        return np.array(None, dtype=object)
    if isinstance(x, dict | Batch):
        y = Batch(x) if isinstance(x, dict) else deepcopy(x)
        y.to_numpy_()
        return y
    if isinstance(x, list | tuple):
        y = to_numpy(_parse_value(x))
        return y
    # fallback
    return np.asanyarray(x)


@overload
def to_torch(
    x: dict | Batch,
    dtype: torch.dtype | None = None,
    device: str | int | torch.device = "cpu",
) -> Batch: ...


@overload
def to_torch(
    x: _NumNDArr | _Number,
    dtype: torch.dtype | None = None,
    device: str | int | torch.device = "cpu",
) -> torch.Tensor: ...


@no_type_check
def to_torch(
    x: Any,
    dtype: torch.dtype | None = None,
    device: str | int | torch.device = "cpu",
) -> Batch | torch.Tensor:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type,
        np.bool_ | np.number,
    ):  # most often case
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    if isinstance(x, np.number | np.bool_ | Number):
        return to_torch(np.asanyarray(x), dtype, device)
    if isinstance(x, dict | Batch):
        x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
        x.to_torch_(dtype, device)
        return x
    if isinstance(x, list | tuple):
        return to_torch(_parse_value(x), dtype, device)
    # fallback
    raise TypeError(f"object {x} cannot be converted to torch.")


@overload
def to_torch_as(x: dict | Batch, y: torch.Tensor) -> Batch: ...


@overload
def to_torch_as(x: _NumNDArr | _Number, y: torch.Tensor) -> torch.Tensor: ...


@no_type_check
def to_torch_as(x: Any, y: torch.Tensor) -> Batch | torch.Tensor:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)
