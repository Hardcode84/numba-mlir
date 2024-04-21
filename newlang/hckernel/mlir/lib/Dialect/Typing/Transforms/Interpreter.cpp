// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

mlir::FailureOr<llvm::SmallVector<mlir::Type>>
hc::typing::Interpreter::run(TypeResolverOp resolver, mlir::TypeRange types) {
  return mlir::failure();
}
