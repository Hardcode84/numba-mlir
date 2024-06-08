# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .kernel_api import _verify_kernel_params
from .kernel_api import *
from .dispatcher import create_dispatcher


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
):
    _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables)

    def _kernel_impl(func):
        return create_dispatcher(func)

    return _kernel_impl
