from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, Union

_T_ArithElem_co = TypeVar("T_ArithElem_co", covariant=True)


class BaseModel(ABC):
    name: str
    """model name"""

    dimX: int
    """number of one-step states"""

    dimU: int
    """number of one-step inputs"""

    @abstractmethod
    def dynamics(
        self,
        t: _T_ArithElem_co,
        X: Sequence[_T_ArithElem_co],
        U: Sequence[_T_ArithElem_co],
    ) -> Sequence[_T_ArithElem_co]:
        raise NotImplementedError
