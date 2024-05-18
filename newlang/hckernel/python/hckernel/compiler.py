# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType
from hckernel._native.compiler import create_context, Dispatcher

mlir_context = create_context()
