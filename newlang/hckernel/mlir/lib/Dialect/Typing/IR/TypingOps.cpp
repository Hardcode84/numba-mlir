// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
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

  registerArithTypingInterpreter(*getContext());
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

namespace {
using namespace hc::typing;
struct ConstantOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          ConstantOpInterpreterInterface, mlir::arith::ConstantOp> {
  mlir::FailureOr<bool> interpret(mlir::Operation *o,
                                  InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::ConstantOp>(o);
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(op.getValue());
    if (!attr)
      return op->emitError("Expected int attribute but got ") << op.getValue();

    state.state[op.getResult()] = setInt(op.getContext(), attr.getInt());
    return true;
  }
};

struct AddOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          AddOpInterpreterInterface, mlir::arith::AddIOp> {
  mlir::FailureOr<bool> interpret(mlir::Operation *o,
                                  InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::AddIOp>(o);
    auto lhs = getInt(state, op.getLhs());
    if (!lhs)
      return op->emitError("Invalid lhs val");

    auto rhs = getInt(state, op.getRhs());
    if (!rhs)
      return op->emitError("Invalid rhs val");

    state.state[op.getResult()] = setInt(op.getContext(), *lhs + *rhs);
    return true;
  }
};

struct CmpOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          CmpOpInterpreterInterface, mlir::arith::CmpIOp> {
  mlir::FailureOr<bool> interpret(mlir::Operation *o,
                                  InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::CmpIOp>(o);
    auto lhs = getInt(state, op.getLhs());
    if (!lhs)
      return op->emitError("Invalid lhs val");

    auto rhs = getInt(state, op.getRhs());
    if (!rhs)
      return op->emitError("Invalid rhs val");

    int64_t res;
    using Pred = mlir::arith::CmpIPredicate;
    switch (op.getPredicate()) {
    case Pred::eq:
      res = (*lhs == *rhs);
      break;
    case Pred::ne:
      res = (*lhs != *rhs);
      break;
    case Pred::slt:
      res = (*lhs < *rhs);
      break;
    case Pred::sle:
      res = (*lhs <= *rhs);
      break;
    case Pred::sgt:
      res = (*lhs > *rhs);
      break;
    case Pred::sge:
      res = (*lhs >= *rhs);
      break;
    case Pred::ult:
      res = (static_cast<uint64_t>(*lhs) < static_cast<uint64_t>(*rhs));
      break;
    case Pred::ule:
      res = (static_cast<uint64_t>(*lhs) <= static_cast<uint64_t>(*rhs));
      break;
    case Pred::ugt:
      res = (static_cast<uint64_t>(*lhs) > static_cast<uint64_t>(*rhs));
      break;
    case Pred::uge:
      res = (static_cast<uint64_t>(*lhs) >= static_cast<uint64_t>(*rhs));
      break;
    default:
      return op->emitError("Unsupported predicate: ") << op.getPredicateAttr();
    }

    state.state[op.getResult()] = setInt(op.getContext(), res);
    return true;
  }
};
} // namespace

void hc::typing::registerArithTypingInterpreter(mlir::MLIRContext &ctx) {
  ctx.loadDialect<mlir::arith::ArithDialect>();

  mlir::arith::ConstantOp::attachInterface<ConstantOpInterpreterInterface>(ctx);
  mlir::arith::AddIOp::attachInterface<AddOpInterpreterInterface>(ctx);
  mlir::arith::CmpIOp::attachInterface<CmpOpInterpreterInterface>(ctx);
}

template <typename Dst, typename Src>
static auto castArrayRef(mlir::ArrayRef<Src> src) {
  return mlir::ArrayRef<Dst>(static_cast<const Dst *>(src.data()), src.size());
}

static const constexpr int PackShift = 2;

std::optional<int64_t> hc::typing::getInt(InterpreterValue val) {
  if (val.is<void *>())
    return reinterpret_cast<intptr_t>(val.get<void *>()) >> PackShift;

  auto lit = mlir::dyn_cast<hc::typing::LiteralType>(val.get<mlir::Type>());
  if (!lit)
    return std::nullopt;

  auto attr = mlir::dyn_cast<mlir::IntegerAttr>(lit.getValue());
  if (!attr)
    return std::nullopt;

  return attr.getInt();
}

std::optional<int64_t> hc::typing::getInt(InterpreterState &state,
                                          mlir::Value val) {
  auto it = state.state.find(val);
  assert(it != state.state.end());
  return getInt(it->second);
}

hc::typing::InterpreterValue hc::typing::setInt(mlir::MLIRContext *ctx,
                                                int64_t val) {
  if (((static_cast<intptr_t>(val) << PackShift) >> PackShift) == val)
    return reinterpret_cast<void *>(static_cast<intptr_t>(val) << PackShift);

  auto attr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), val);
  return hc::typing::LiteralType::get(attr);
}

mlir::Type hc::typing::getType(const hc::typing::InterpreterState &state,
                               mlir::Value val) {
  auto it = state.state.find(val);
  assert(it != state.state.end());
  return mlir::cast<mlir::Type>(it->second);
}

void hc::typing::getTypes(const hc::typing::InterpreterState &state,
                          mlir::ValueRange vals,
                          llvm::SmallVectorImpl<mlir::Type> &result) {
  result.reserve(result.size() + vals.size());
  for (auto val : vals)
    result.emplace_back(getType(state, val));
}

llvm::SmallVector<mlir::Type>
hc::typing::getTypes(const hc::typing::InterpreterState &state,
                     mlir::ValueRange vals) {
  llvm::SmallVector<mlir::Type> ret;
  getTypes(state, vals, ret);
  return ret;
}

mlir::FailureOr<bool>
hc::typing::MakeIdent::interpret(InterpreterState &state) {
  auto name = this->getNameAttr();
  auto paramNames =
      castArrayRef<mlir::StringAttr>(this->getParamNames().getValue());
  auto paramTypes = getTypes(state, this->getParams());
  state.state[getResult()] = hc::typing::IdentType::get(
      this->getContext(), name, paramNames, paramTypes);
  return true;
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
