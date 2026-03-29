#!/usr/bin/env python
# coding: utf-8
# 250426
"""
this is dot style dictionary like JavaScript.
You can access via both bracket style and dot nonation style.

[example]

from dotdict import dotdict

d = dotdict({'a': 'spam', 'b': 'ham'})
print d['a']
>   spam
print d.b
>   ham
d.c = 'python!'
print d.c
>   python!
"""

__author__ = "vplus"
__license__ = "Public Domain"
__version__ = "0.2"


from ast import alias
from typing import Any, Dict, Generic, OrderedDict, TypeVar, Union

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class DotDict(Dict[Union[_KT, str], _VT]):
    """dict with dot notation access."""

    def __setattr__(self, key: Union[_KT, str], value: _VT):
        super().__setitem__(key, value)

    def __getattr__(self, key: Union[_KT, str]) -> _VT:
        return super().__getitem__(key)

    def __delattr__(self, key):
        super().__delattr__(key)


class dotdict(DotDict[_KT, _VT]):
    """dict with dot notation access."""
    pass



class OrderedDotDict(OrderedDict[Union[_KT, str], _VT]):
    """OrderedDict with dot notation access."""

    def __setattr__(self, key: Union[_KT, str], value: _VT):
        super().__setitem__(key, value)

    def __getattr__(self, key: Union[_KT, str]) -> _VT:
        return super().__getitem__(key)

    def __delattr__(self, key):
        super().__delattr__(key)
