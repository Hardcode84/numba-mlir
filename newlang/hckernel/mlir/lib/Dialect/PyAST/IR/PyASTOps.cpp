// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hckernel/Dialect/PyAST/IR/PyASTOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hckernel::py_ast::PyASTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hckernel/Dialect/PyAST/IR/PyASTOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hckernel/Dialect/PyAST/IR/PyASTOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hckernel/Dialect/PyAST/IR/PyASTOpsAttributes.cpp.inc"
      >();
}

void hckernel::py_ast::PyModuleOp::build(::mlir::OpBuilder &odsBuilder,
                                         ::mlir::OperationState &odsState) {
  ensureTerminator(*odsState.addRegion(), odsBuilder, odsState.location);
}

void hckernel::py_ast::PyFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                       ::mlir::OperationState &odsState,
                                       mlir::ValueRange args,
                                       mlir::ValueRange decorators) {
  odsState.addOperands(args);
  odsState.addOperands(decorators);

  int32_t segmentSizes[2] = {};
  segmentSizes[0] = static_cast<int32_t>(args.size());
  segmentSizes[1] = static_cast<int32_t>(decorators.size());
  odsState.addAttribute(getOperandSegmentSizeAttr(),
                        odsBuilder.getDenseI32ArrayAttr(segmentSizes));

  ensureTerminator(*odsState.addRegion(), odsBuilder, odsState.location);
}

void hckernel::py_ast::ArgOp::build(::mlir::OpBuilder &odsBuilder,
                                    ::mlir::OperationState &odsState,
                                    llvm::StringRef name,
                                    mlir::Value annotation) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, name, annotation);
}

void hckernel::py_ast::NameOp::build(::mlir::OpBuilder &odsBuilder,
                                     ::mlir::OperationState &odsState,
                                     llvm::StringRef id) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, id);
}

void hckernel::py_ast::SubscriptOp::build(::mlir::OpBuilder &odsBuilder,
                                          ::mlir::OperationState &odsState,
                                          mlir::Value value,
                                          mlir::Value slice) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, value, slice);
}

void hckernel::py_ast::TupleOp::build(::mlir::OpBuilder &odsBuilder,
                                      ::mlir::OperationState &odsState,
                                      mlir::ValueRange elts) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, elts);
}

void hckernel::py_ast::AttributeOp::build(::mlir::OpBuilder &odsBuilder,
                                          ::mlir::OperationState &odsState,
                                          mlir::Value value,
                                          llvm::StringRef attr) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, value, attr);
}

void hckernel::py_ast::ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                                         ::mlir::OperationState &odsState,
                                         mlir::Attribute value) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, value);
}

void hckernel::py_ast::SliceOp::build(::mlir::OpBuilder &odsBuilder,
                                      ::mlir::OperationState &odsState,
                                      mlir::Value lower, mlir::Value upper,
                                      mlir::Value step) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, lower, upper, step);
}

void hckernel::py_ast::CallOp::build(::mlir::OpBuilder &odsBuilder,
                                     ::mlir::OperationState &odsState,
                                     mlir::Value func, mlir::ValueRange args,
                                     mlir::ValueRange keywods) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, func, args, keywods);
}

void hckernel::py_ast::KeywordOp::build(::mlir::OpBuilder &odsBuilder,
                                        ::mlir::OperationState &odsState,
                                        llvm::StringRef arg,
                                        mlir::Value value) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, arg, value);
}

void hckernel::py_ast::BoolOp::build(::mlir::OpBuilder &odsBuilder,
                                     ::mlir::OperationState &odsState,
                                     BoolOpType op, mlir::ValueRange values) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, op, values);
}

void hckernel::py_ast::IfOp::build(::mlir::OpBuilder &odsBuilder,
                                   ::mlir::OperationState &odsState,
                                   mlir::Value test, bool hasElse) {
  odsState.addOperands(test);
  auto thenRegion = odsState.addRegion();
  auto elseRegion = odsState.addRegion();
  ensureTerminator(*thenRegion, odsBuilder, odsState.location);
  if (hasElse)
    ensureTerminator(*elseRegion, odsBuilder, odsState.location);
}

void hckernel::py_ast::CompareOp::build(::mlir::OpBuilder &odsBuilder,
                                        ::mlir::OperationState &odsState,
                                        mlir::Value left,
                                        mlir::ArrayRef<CmpOp> ops,
                                        mlir::ValueRange comparators) {
  auto type = NodeType::get(odsBuilder.getContext());
  llvm::SmallVector<mlir::Attribute> opAttrs;
  opAttrs.reserve(ops.size());
  for (auto op : ops)
    opAttrs.emplace_back(CmpOpAttr::get(odsBuilder.getContext(), op));

  build(odsBuilder, odsState, type, left, odsBuilder.getArrayAttr(opAttrs),
        comparators);
}

void hckernel::py_ast::BinOp::build(::mlir::OpBuilder &odsBuilder,
                                    ::mlir::OperationState &odsState,
                                    mlir::Value left, BinOpVal op,
                                    mlir::Value right) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, left, op, right);
}

#include "hckernel/Dialect/PyAST/IR/PyASTOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hckernel/Dialect/PyAST/IR/PyASTOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hckernel/Dialect/PyAST/IR/PyASTOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hckernel/Dialect/PyAST/IR/PyASTOpsTypes.cpp.inc"

#include "hckernel/Dialect/PyAST/IR/PyASTOpsEnums.cpp.inc"
