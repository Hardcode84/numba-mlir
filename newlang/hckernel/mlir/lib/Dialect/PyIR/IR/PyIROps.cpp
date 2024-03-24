// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::py_ir::PyIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"
      >();
}

void hc::py_ir::PyFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                llvm::StringRef name, mlir::TypeRange argTypes,
                                mlir::ValueRange decorators) {
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  odsState.addOperands(decorators);

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);
  odsBuilder.createBlock(region, {}, argTypes);
}

#include "hc/Dialect/PyIR/IR/PyIROpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"

#include "hc/Dialect/PyIR/IR/PyIROpsEnums.cpp.inc"
