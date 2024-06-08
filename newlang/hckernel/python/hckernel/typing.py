# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType
from hckernel._native.compiler import (
    TypingDispatcher,
    MlirDecoratorWrapper,
    i1,
    check,
    is_same,
)
from hckernel.compiler import mlir_context
from hckernel.kernel import FuncDesc


def _get_desc(func):
    if not isinstance(func, FunctionType):
        raise RuntimeError(f"Unsupported object {type(func)}")

    def _wrapper():
        sig = inspect.signature(func)

        return FuncDesc(
            source=inspect.getsource(func),
            name=func.__name__,
            args=None,
            imported_symbols=None,
        )

    return _wrapper


class Resolver(MlirDecoratorWrapper):
    def __call__(self, func):
        return TypingDispatcher(mlir_context, _get_desc(func), func.__globals__)


resolver = Resolver()
