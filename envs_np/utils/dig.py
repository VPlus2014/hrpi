from __future__ import annotations
from typing import Any, Generic, TypeVar
from types import GenericAlias
from typing import Callable

_OFF = False  # 关闭. 建议内存充足时开启
_PREFIX = "__DIG__"
_NOT_FOUND = object()

_T = TypeVar("_T")


class DIG_property(Generic[_T]):
    def __init__(self, func: Callable[[Any], _T]):
        self.func = func
        self.attrname = None
        self._bufname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: type, name: str):
        if self.attrname is None:
            self.attrname = name
            self._bufname = _PREFIX + name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance: object, owner: type | None = None) -> _T:
        if _OFF:
            return self.func(instance)
        if instance is None:
            raise TypeError(f"error use NoneType.{self.attrname}")
        attn = self._bufname
        if attn is None:
            raise TypeError(
                "Cannot use DIG_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {attn!r} property."
            )
            raise TypeError(msg) from None

        val: _T = cache.get(attn, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[attn] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.attrname!r} property"
                    f"with attribute name: {attn!r}"
                )
                raise TypeError(msg) from None
        return val

    __class_getitem__ = classmethod(GenericAlias)


def DIG_clean(obj: object):
    """
    Dynamic Information Graph 相关的变量清理(谨慎使用, 谨防死锁)

    TODO: 建立依赖关系图, 允许只清除部分节点
    """
    attns = []
    for attr in obj.__dict__:
        if attr.startswith(_PREFIX):
            attns.append(attr)
    for attr in attns:
        delattr(obj, attr)
