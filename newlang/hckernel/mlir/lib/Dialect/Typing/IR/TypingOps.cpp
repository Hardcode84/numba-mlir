// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::typing::TypingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/Typing/IR/TypingOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/Typing/IR/TypingOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/Typing/IR/TypingOpsAttributes.cpp.inc"
      >();
}

#include "hc/Dialect/Typing/IR/TypingOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/Typing/IR/TypingOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsTypes.cpp.inc"

#include "hc/Dialect/Typing/IR/TypingOpsEnums.cpp.inc"
