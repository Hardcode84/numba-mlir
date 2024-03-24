// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include <mlir/Bytecode/BytecodeOpInterface.h>

#include "hc/Dialect/PyIR/IR/PyIROpsDialect.h.inc"
#include "hc/Dialect/PyIR/IR/PyIROpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROps.h.inc"
