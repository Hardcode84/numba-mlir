# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .typing import type_resolver, func, TypingRegistry

_registry = TypingRegistry()


def get_typing_module():
    global _registry
    _registry.compile_type_resolvers()
    return _registry.module
