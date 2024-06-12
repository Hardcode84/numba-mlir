# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_registry import register_symbol as _reg_symbol_impl
from .dispatcher import create_dispatcher
from ._native.compiler._typing import TypingDispatcher, link_modules, load_mlir_module
from .bitcode_storage import get_bitcode_file
from .mlir import ir


def _register_symbol(sym):
    _reg_symbol_impl(eval(sym), sym, __name__)


def _stub_error():
    raise NotImplementedError("This is a stub")


_typing_prelink = load_mlir_module(ir._BaseContext.current, get_bitcode_file("typing"))

_typing_dispatchers = []


def type_resolver(key):
    def _wrapper(func):
        disp = create_dispatcher(
            func, prelink_module=_typing_prelink, dispatcher=TypingDispatcher
        )
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


def is_same(a, b):
    _stub_error()


def check(cond):
    _stub_error()


_register_symbol("type_resolver")
_register_symbol("is_same")
_register_symbol("check")
