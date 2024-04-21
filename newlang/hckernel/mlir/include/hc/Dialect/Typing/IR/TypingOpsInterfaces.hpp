// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>

namespace hc::typing {
using InterpreterState = llvm::DenseMap<mlir::Value, mlir::Type>;
mlir::Type getType(const hc::typing::InterpreterState &state, mlir::Value val);

llvm::SmallVector<mlir::Type>
getTypes(const hc::typing::InterpreterState &state, mlir::ValueRange vals);
} // namespace hc::typing

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.h.inc"
