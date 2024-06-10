# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_registry import register_symbol as _reg_symbol_impl
from .dispatcher import create_dispatcher
from ._native.compiler._typing import TypingDispatcher


def _register_symbol(sym):
    _reg_symbol_impl(eval(sym), sym, __name__)


_typing_dispatchers = []


def type_resolver(key):
    def _wrapper(func):
        disp = create_dispatcher(func, TypingDispatcher)
        _typing_dispatchers.append(disp)
        return disp

    return _wrapper


_register_symbol("type_resolver")
