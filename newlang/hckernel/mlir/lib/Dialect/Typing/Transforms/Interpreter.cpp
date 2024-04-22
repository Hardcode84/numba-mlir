// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

static mlir::FailureOr<bool> handleOp(hc::typing::InterpreterState &state,
                                      mlir::Operation &op) {
  if (auto iface = mlir::dyn_cast<hc::typing::TypingInterpreterInterface>(op))
    return iface.interpret(state);

  return op.emitError("Type interpreter: unsupported op");
}

mlir::FailureOr<llvm::SmallVector<mlir::Type>>
hc::typing::Interpreter::run(TypeResolverOp resolver, mlir::TypeRange types) {
  state.clear();
  assert(!resolver.getBodyRegion().empty());
  mlir::Block *block = &resolver.getBodyRegion().front();
  while (true) {
    for (mlir::Operation &op : block->without_terminator()) {
      auto res = handleOp(state, op);
      if (mlir::failed(res))
        return mlir::failure();

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
