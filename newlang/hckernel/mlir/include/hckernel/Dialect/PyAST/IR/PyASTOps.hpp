// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

//#include <mlir/Interfaces/CastInterfaces.h>
//#include <mlir/Interfaces/ControlFlowInterfaces.h>
//#include <mlir/Interfaces/CopyOpInterface.h>
//#include <mlir/Interfaces/InferTypeOpInterface.h>
//#include <mlir/Interfaces/ShapedOpInterfaces.h>
//#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

namespace hckernel::py_ast {
template <typename ConcreteType>
class NoReturn : public mlir::OpTrait::TraitBase<ConcreteType, NoReturn> {};
} // namespace hckernel::py_ast

#include "hckernel/Dialect/PyAST/IR/PyASTOpsDialect.h.inc"
#include "hckernel/Dialect/PyAST/IR/PyASTOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hckernel/Dialect/PyAST/IR/PyASTOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hckernel/Dialect/PyAST/IR/PyASTOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hckernel/Dialect/PyAST/IR/PyASTOps.h.inc"
