// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace llvm {
class StringRef;
}

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace hc {
mlir::LogicalResult importPyModule(llvm::StringRef str, mlir::ModuleOp module);
}
