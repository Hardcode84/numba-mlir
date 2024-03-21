// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <hckernel/Dialect/PyAST/IR/PyASTOps.hpp>

namespace hckernel {
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<hckernel::py_ast::PyASTDialect>();
}

inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  ::hckernel::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace hckernel
