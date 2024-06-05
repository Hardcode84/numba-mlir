# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType
from collections import namedtuple, OrderedDict

from .kernel_api import _verify_kernel_params
from .kernel_api import *
from .compiler import mlir_context, Dispatcher
from .symbol_registry import get_module_for_symbol


FuncDesc = namedtuple("FuncDesc", ["source", "name", "args", "imported_symbols"])


def _process_annotation(ann):
    def istypingtype(a, typ):
        return typing.get_origin(ann) == typ or isinstance(ann, typ)

    def get_typing_args(ann):
        if isinstance(ann, (types.GenericAlias, typing._GenericAlias)):
            return typing.get_args(ann)[0]

        if isinstance(ann, Iterable):
            return ann

        assert False

    if ann in (CurrentGroup, CurrentSubGroup, CurrentWorkitem):
        # nothing
        return

    if isinstance(ann, Symbol):
        return "sym"

    elif istypingtype(ann, tuple):
        return tuple(_process_annotation(e) for e in get_typing_args(ann))

    elif istypingtype(ann, Buffer):
        return [_process_annotation(e) for e in get_typing_args(ann)]

    else:
        assert False, f"Unsupported annotation: {type(ann)} {ann}"


def _get_desc(func):
    if not isinstance(func, FunctionType):
        raise RuntimeError(f"Unsupported object {type(func)}")

    def _wrapper():
        sig = inspect.signature(func)
        args_types = OrderedDict()
        for name, param in sig.parameters.items():
            annotation = param.annotation
            assert annotation != param.empty
            annotation = _process_annotation(annotation)
            if annotation is None:
                continue

            args_types[name] = annotation

        imported_symbols = []
        for name, obj in func.__globals__.items():
            mod = get_module_for_symbol(obj)
            if not mod:
                continue
            imported_symbols.append((name, mod))

        return FuncDesc(
            source=inspect.getsource(func),
            name=func.__name__,
            args=args_types,
            imported_symbols=imported_symbols,
        )

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
