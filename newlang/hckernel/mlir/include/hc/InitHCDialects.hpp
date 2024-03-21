// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <hc/Dialect/PyAST/IR/PyASTOps.hpp>

namespace hc {
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<hc::py_ast::PyASTDialect>();
}

inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  ::hc::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace hc
