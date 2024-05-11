import inspect
from types import FunctionType
from hckernel._native.compiler import compile_ast


def mlir_compile(func):
    if isinstance(func, str):
        return compile_ast(func)

    if isinstance(func, FunctionType):
        return compile_ast(inspect.getsource(func))

    raise RuntimeError(f"Unsupported object {type(func)}")
