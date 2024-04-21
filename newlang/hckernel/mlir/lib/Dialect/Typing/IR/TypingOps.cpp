// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

namespace {
struct TypingAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Type type, llvm::raw_ostream &os) const final {
    if (auto ident = llvm::dyn_cast<hc::typing::IdentType>(type)) {
      os << "ident";
      return AliasResult::OverridableAlias;
    }
    if (auto seq = llvm::dyn_cast<hc::typing::SequenceType>(type)) {
      os << "seq";
      return AliasResult::OverridableAlias;
    }
    if (auto sym = llvm::dyn_cast<hc::typing::SymbolType>(type)) {
      os << "sym";
      return AliasResult::OverridableAlias;
    }
    if (auto lit = llvm::dyn_cast<hc::typing::LiteralType>(type)) {
      os << "literal";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

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

  addInterface<TypingAsmDialectInterface>();
}

void hc::typing::ResolveOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  mlir::TypeRange resultTypes,
                                  mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(resultTypes);

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);

  llvm::SmallVector<mlir::Location> locs(args.size(),
                                         odsBuilder.getUnknownLoc());
  odsBuilder.createBlock(region, {}, mlir::TypeRange(args), locs);
}

#include "hc/Dialect/Typing/IR/TypingOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/Typing/IR/TypingOps.cpp.inc"

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsTypes.cpp.inc"

#include "hc/Dialect/Typing/IR/TypingOpsEnums.cpp.inc"
