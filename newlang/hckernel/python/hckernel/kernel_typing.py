# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .typing import *
from .mlir import typing

_registry = TypingRegistry()


def get_typing_module():
    global _registry
    _registry.compile_type_resolvers()
    return _registry.module


ValueType = typing.ValueType.get()
HCKernelMod = typing.IdentType.get("hckernel")
HCKernelAPI = typing.IdentType.get("hckernel.kernel_api")
Indexing = typing.IdentType.get("hckernel.indexing")
BufferBase = typing.IdentType.get("hckernel.kernel_api.BufferBase")
CurrentGroup = typing.IdentType.get("hckernel.kernel_api.CurrentGroup")


@func
def check_type(a: ValueType, b: ValueType):
    check(is_same(a, b))


@type_resolver(_registry, ["py_ir.load_module", "hckernel"])
def resolver():
    return HCKernelMod


@type_resolver(_registry, ["py_ir.getattr", "kernel_api"])
def resolver(a: ValueType):
    check_type(a, HCKernelMod)
    return HCKernelAPI


@type_resolver(_registry, ["py_ir.getattr", "indexing"])
def resolver(a: ValueType):
    check_type(a, HCKernelMod)
    return Indexing


@type_resolver(_registry, ["py_ir.getattr"])
def resolver(a: ValueType):
    check_type(a, Indexing)
    return make_symbol(get_attr("name"))


@type_resolver(_registry, ["py_ir.getattr", "Buffer"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return BufferBase


@type_resolver(_registry, ["py_ir.getattr", "CurrentGroup"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return CurrentGroup
