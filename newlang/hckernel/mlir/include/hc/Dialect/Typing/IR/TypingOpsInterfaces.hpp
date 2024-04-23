// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OpDefinition.h>

namespace hc::typing {
void registerArithTypingInterpreter(mlir::MLIRContext &ctx);

using InterpreterValue = llvm::PointerUnion<mlir::Type, void *>;

struct InterpreterState {
  void init(mlir::Block &block, mlir::TypeRange types) {
    state.clear();
    args = types;
    iter = block.begin();
    completed = false;
  }

  mlir::Operation &getNextOp() {
    auto it = iter++;
    return *it;
  }

  void setCompleted() { completed = true; }
  bool isCompleted() const { return completed; }

  llvm::DenseMap<mlir::Value, InterpreterValue> state;
  mlir::TypeRange args;
  mlir::Block::iterator iter;
  bool completed = false;
};

std::optional<int64_t> getInt(InterpreterValue val);
std::optional<int64_t> getInt(InterpreterState &state, mlir::Value val);

InterpreterValue setInt(mlir::MLIRContext *ctx, int64_t val);

mlir::Type getType(const hc::typing::InterpreterState &state, mlir::Value val);

void getTypes(const hc::typing::InterpreterState &state, mlir::ValueRange vals,
              llvm::SmallVectorImpl<mlir::Type> &result);

llvm::SmallVector<mlir::Type>
getTypes(const hc::typing::InterpreterState &state, mlir::ValueRange vals);
} // namespace hc::typing

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.h.inc"
