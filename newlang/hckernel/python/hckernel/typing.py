# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_registry import register_symbol as _reg_symbol_impl
from .dispatcher import create_dispatcher


def _register_symbol(sym):
    _reg_symbol_impl(eval(sym), sym, __name__)


def type_resolver(key):
    def _wrapper(func):
        return create_dispatcher(func)

    return _wrapper


_register_symbol("type_resolver")
