# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .kernel_api import _verify_kernel_params
from .kernel_api import *
from .dispatcher import create_dispatcher


def _get_num_dims(arg):
    if not isinstance(arg, Iterable):
        return 1

    return len(arg)


def _resolve_globals(func, mapping):
    old_closure = func.__closure__
    new_closure = None

    def resolve_mapping(val):
        try:
            return mapping[val]
        except:
            pass

        return None

    if old_closure is not None:
        cell_cls = type(old_closure[0])

        def resolve_cell(cell):
            new_val = resolve_mapping(cell.cell_contents)
            if new_val is not None:
                return cell_cls(new_val)

            return cell

        new_closure = tuple([resolve_cell(cell) for cell in old_closure])

    def resolve_global(val):
        new_val = resolve_mapping(val)
        if new_val is not None:
            return new_val

        return val

    new_globals = {key: resolve_global(val) for key, val in func.__globals__.items()}

    g = types.FunctionType(
        func.__code__,
        new_globals,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=new_closure,
    )
    g = functools.update_wrapper(g, func)
    g.__kwdefaults__ = func.__kwdefaults__
    return g


def _get_typing_module():
    from .kernel_typing import get_typing_module

    return get_typing_module()


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
):
    _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables)

    def _kernel_impl(func):
        gr_index = _get_num_dims(work_shape) - 1
        new_current_group = [CurrentGroup1, CurrentGroup2, CurrentGroup3][gr_index]
        mapping = {CurrentGroup: new_current_group}
        new_func = _resolve_globals(func, mapping)
        return create_dispatcher(new_func, prelink_module=_get_typing_module)

    return _kernel_impl
