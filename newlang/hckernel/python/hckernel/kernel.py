# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .kernel_api import _verify_kernel_params
from .kernel_api import *

from hckernel._native.compiler import Dispatcher


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
):
    _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables)

    def _kernel_impl(func):
        return Dispatcher()

    return _kernel_impl
