// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class LogicalResult;
class ModuleOp;
} // namespace mlir

namespace hc {
mlir::LogicalResult linkModules(mlir::ModuleOp dest, mlir::ModuleOp toLink);
}
