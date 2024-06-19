# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .typing import type_resolver, func, is_same, check, TypingRegistry
from .mlir import typing

_registry = TypingRegistry()


def get_typing_module():
    global _registry
    _registry.compile_type_resolvers()
    return _registry.module


ValueType = typing.ValueType.get()
HCKernelMod = typing.IdentType.get("hckernel")
HCKernelAPI = typing.IdentType.get("hckernel.kernel_api")


@func
def check_type(a: ValueType, b: ValueType):
    check(is_same(a, b))


@type_resolver(_registry, ["py_ir.load_module", "hckernel"])
def module_resolver():
    return HCKernelMod


@type_resolver(_registry, ["py_ir.getattr", "kernel_api"])
def kernel_api_resolver(a: ValueType):
    check_type(a, HCKernelMod)
    return HCKernelAPI
