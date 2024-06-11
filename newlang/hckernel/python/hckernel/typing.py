# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_registry import register_symbol as _reg_symbol_impl
from .dispatcher import create_dispatcher
from ._native.compiler._typing import TypingDispatcher, link_modules


def _register_symbol(sym):
    _reg_symbol_impl(eval(sym), sym, __name__)


_typing_dispatchers = []


def type_resolver(key):
    def _wrapper(func):
        disp = create_dispatcher(func, TypingDispatcher)
        _typing_dispatchers.append(disp)
        return disp

    return _wrapper


_typing_module = None


def compile_type_resolvers():
    global _typing_dispatchers
    global _typing_module
    try:
        for disp in _typing_dispatchers:
            mod = disp.compile()
            if _typing_module is None:
                _typing_module = mod
            else:
                link_modules(_typing_module, mod)

    finally:
        # To properly handle exceptions in `compile()`
        _typing_dispatchers.clear()


_register_symbol("type_resolver")
