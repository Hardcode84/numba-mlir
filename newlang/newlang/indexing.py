
from typing import Any, ClassVar, Optional, Type, TypeVar, Union, cast
import sympy

IndexSymbol = sympy.core.Symbol
IndexExpr = sympy.core.Expr

def index_symbol(name: str) -> IndexSymbol:
    return sympy.Symbol(name, integer=True, nonnegative=True)


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n):
        return index_symbol(n)

sym = _IndexSymbolExpando()
