# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType

from .kernel_api import _verify_kernel_params
from .kernel_api import *
from .compiler import mlir_context, Dispatcher


def _get_source(func):
    def _wrapper():
        return inspect.getsource(func), func.__name__

    return _wrapper


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
):
    _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables)

    def _kernel_impl(func):
        if not isinstance(func, FunctionType):
            raise RuntimeError(f"Unsupported object {type(func)}")

        return Dispatcher(mlir_context, _get_source(func))

    return _kernel_impl
