# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, ClassVar, Optional, Type, TypeVar, Union, cast
from .symbol_registry import register_symbol as _reg_symbol_impl
import sympy

IndexSymbol = sympy.core.Symbol
IndexExpr = sympy.core.Expr


def index_symbol(name: str) -> IndexSymbol:
    return sympy.Symbol(name, integer=True)


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n):
        s = index_symbol(n)
        _reg_symbol_impl(s, n, __name__)
        return s


sym = _IndexSymbolExpando()
