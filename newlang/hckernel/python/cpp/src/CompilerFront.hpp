// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

namespace llvm {
class StringRef;
}

namespace mlir {
class MLIRContext;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::Operation *>>
compileAST(mlir::MLIRContext &ctx, llvm::StringRef source,
           llvm::StringRef funcName);
