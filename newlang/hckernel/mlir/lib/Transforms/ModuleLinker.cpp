// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ModuleLinker.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>

mlir::LogicalResult hc::linkModules(mlir::ModuleOp dest,
                                    mlir::ModuleOp toLink) {
  // TODO: actually resolve symbols
  auto builder = mlir::OpBuilder::atBlockEnd(dest.getBody());
  mlir::IRMapping mapper;
  for (mlir::Operation &op : toLink.getBody()->getOperations())
    builder.clone(op, mapper);

  return mlir::success();
}
