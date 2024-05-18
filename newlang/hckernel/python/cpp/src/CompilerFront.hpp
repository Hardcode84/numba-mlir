// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace llvm {
class StringRef;
}

namespace mlir {
class MLIRContext;
struct LogicalResult;
} // namespace mlir

mlir::LogicalResult compileAST(mlir::MLIRContext &ctx, llvm::StringRef source,
                               llvm::StringRef funcName);
