#pragma once

namespace llvm {
class StringRef;
}

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace hckernel {
mlir::LogicalResult importPyModule(llvm::StringRef str, mlir::ModuleOp module);
}
