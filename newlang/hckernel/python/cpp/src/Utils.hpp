// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace llvm {
class Twine;
}

namespace mlir {
class Operation;
class PassManager;
struct LogicalResult;
} // namespace mlir

[[noreturn]] void reportError(const llvm::Twine &msg);

mlir::LogicalResult runUnderDiag(mlir::PassManager &pm,
                                 mlir::Operation *module);
