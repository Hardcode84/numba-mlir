// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

using State = llvm::DenseMap<mlir::Value, mlir::Type>;

static mlir::Type getType(const State &state, mlir::Value val) {
  auto it = state.find(val);
  assert(it != state.end());
  return it->second;
}

static llvm::SmallVector<mlir::Type> getTypes(const State &state,
                                              mlir::ValueRange vals) {
  llvm::SmallVector<mlir::Type> ret(vals.size());
  for (auto &&[i, val] : llvm::enumerate(vals))
    ret[i] = getType(state, val);

  return ret;
}

static bool handleOp(State &state, mlir::Operation &op) {
  if (auto iface = mlir::dyn_cast<hc::typing::TypingInterpreterInterface>(op))
    return mlir::succeeded(iface.interpret(state));

  return false;
}

mlir::FailureOr<llvm::SmallVector<mlir::Type>>
hc::typing::Interpreter::run(TypeResolverOp resolver, mlir::TypeRange types) {
  state.clear();
  assert(!resolver.getBodyRegion().empty());
  mlir::Block *block = &resolver.getBodyRegion().front();
  while (true) {
    for (mlir::Operation &op : block->without_terminator()) {
      if (!handleOp(state, op))
        return op.emitError("Type interpreter: unsupported op");

      auto term = block->getTerminator();
      if (auto ret = mlir::dyn_cast<TypeResolverReturnOp>(term)) {
        return getTypes(state, ret.getArgs());
      } else {
        return term->emitError("Unsupported terminator");
      }
    }
  }
  llvm_unreachable("Unreachable");
}
