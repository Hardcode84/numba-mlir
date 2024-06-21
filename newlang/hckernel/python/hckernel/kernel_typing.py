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
TupleBase = typing.IdentType.get("Tuple")


@func
def check_type(a: ValueType, b: ValueType):
    check(is_same(a, b))


@func
def check_is_tuple(t: ValueType):
    base_type = make_type(get_type_name(t))
    check(is_same(base_type, TupleBase))


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


@type_resolver(_registry, ["py_ir.tuple_pack"])
def resolver(a: ValueType):
    count = get_num_args()
    i = 0
    seq = create_seq()
    while i < count:
        seq = append_seq(seq, get_arg(i))
        i += 1

    return make_type("Tuple", elements=seq)


@type_resolver(_registry, ["py_ir.getitem"])
def resolver(target: ValueType, index: ValueType):
    check_type(target, BufferBase)
    # check_is_tuple(index)
    elements = get_type_param(index, "elements")
    return make_type("Buffer", dims=elements)


@type_resolver(_registry, ["py_ir.getattr", "Buffer"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return BufferBase


@type_resolver(_registry, ["py_ir.getattr", "CurrentGroup"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return CurrentGroup
