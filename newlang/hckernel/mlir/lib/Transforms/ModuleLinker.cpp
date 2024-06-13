// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ModuleLinker.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>

mlir::LogicalResult hc::linkModules(mlir::Operation *dest,
                                    mlir::Operation *toLink) {
  if (dest->getName() != toLink->getName())
    return dest->emitOpError("Incompatible op types");

  if (dest->getNumRegions() != 1 || !llvm::hasSingleElement(dest->getRegion(0)))
    return dest->emitOpError("Expected 1 region with 1 block");

  if (toLink->getNumRegions() != 1 ||
      !llvm::hasSingleElement(toLink->getRegion(0)))
    return toLink->emitOpError("Expected 1 region with 1 block");

  // TODO: actually resolve symbols
  mlir::Block &dstBlock = dest->getRegion(0).front();
  if (dstBlock.mightHaveTerminator()) {
    mlir::Operation *term = dstBlock.getTerminator();
    if (term->getNumOperands() != 0)
      return dest->emitError("Non-trivial terminators are not supported");

    term->erase();
  }

  mlir::Block &toLinkBlock = toLink->getRegion(0).front();
  if (toLinkBlock.mightHaveTerminator()) {
    mlir::Operation *term = toLinkBlock.getTerminator();
    if (term->getNumOperands() != 0)
      return dest->emitError("Non-trivial terminators are not supported");
  }

  auto builder = mlir::OpBuilder::atBlockEnd(&dstBlock);
  mlir::IRMapping mapper;
  for (mlir::Operation &op : toLinkBlock.getOperations())
    builder.clone(op, mapper);

  return mlir::success();
}
