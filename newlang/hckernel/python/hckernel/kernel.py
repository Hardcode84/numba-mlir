# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType
from collections import namedtuple

from .kernel_api import _verify_kernel_params
from .kernel_api import *
from .compiler import mlir_context, Dispatcher


FuncDesc = namedtuple("FuncDesc", ["source", "name"])


def _get_desc(func):
    if not isinstance(func, FunctionType):
        raise RuntimeError(f"Unsupported object {type(func)}")

    def _wrapper():
        return FuncDesc(source=inspect.getsource(func), name=func.__name__)

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
        return Dispatcher(mlir_context, _get_desc(func))

    return _kernel_impl
