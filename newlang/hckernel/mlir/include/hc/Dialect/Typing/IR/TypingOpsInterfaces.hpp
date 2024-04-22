// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>

namespace hc::typing {
using InterpreterValue = llvm::PointerUnion<mlir::Type, void *>;

std::optional<int64_t> getInt(InterpreterValue val);

InterpreterValue setInt(mlir::MLIRContext *ctx, int64_t val);

using InterpreterState = llvm::DenseMap<mlir::Value, InterpreterValue>;

mlir::Type getType(const hc::typing::InterpreterState &state, mlir::Value val);

void getTypes(const hc::typing::InterpreterState &state, mlir::ValueRange vals,
              llvm::SmallVectorImpl<mlir::Type> &result);

llvm::SmallVector<mlir::Type>
getTypes(const hc::typing::InterpreterState &state, mlir::ValueRange vals);
} // namespace hc::typing

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.h.inc"
