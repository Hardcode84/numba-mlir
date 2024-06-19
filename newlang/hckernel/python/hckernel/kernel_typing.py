# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .typing import type_resolver, func, TypingRegistry
from .mlir import typing

_registry = TypingRegistry()


def get_typing_module():
    global _registry
    _registry.compile_type_resolvers()
    return _registry.module


_hckernel = typing.IdentType.get("hckernel")


@type_resolver(_registry, ["py_ir.load_module", "hckernel"])
def module_resolver():
    return _hckernel
