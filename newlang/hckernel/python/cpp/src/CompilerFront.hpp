// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

namespace llvm {
class StringRef;
}

struct Context;

mlir::FailureOr<mlir::OwningOpRef<mlir::Operation *>>
compileAST(Context &ctx, llvm::StringRef source, llvm::StringRef funcName);
