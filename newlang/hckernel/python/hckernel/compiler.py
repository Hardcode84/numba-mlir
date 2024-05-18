# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType
from hckernel._native.compiler import compile_ast, create_context, Dispatcher

mlir_context = create_context()


def mlir_compile(func):
    if not isinstance(func, FunctionType):
        raise RuntimeError(f"Unsupported object {type(func)}")

    return compile_ast(inspect.getsource(func), func.__name__)
