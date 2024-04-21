// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

namespace hc::typing {
class Interpreter final {
public:
  mlir::FailureOr<llvm::SmallVector<mlir::Type>> run(TypeResolverOp resolver,
                                                     mlir::TypeRange types);
};
} // namespace hc::typing
