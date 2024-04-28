// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

static mlir::LogicalResult jumpToBlock(mlir::Operation *op,
                                       InterpreterState &state,
                                       mlir::Block *newBlock,
                                       mlir::ValueRange args) {
  if (newBlock->getNumArguments() != args.size())
    return op->emitError("Block arg count mismatch");

  state.iter = newBlock->begin();

  // Make a temp copy so we won't overwrite values prematurely if we jump to the
  // same block.
  llvm::SmallVector<InterpreterValue> newValues(args.size());
  for (auto &&[i, arg] : llvm::enumerate(args))
    newValues[i] = state.state[arg];

  for (auto &&[i, arg] : llvm::enumerate(newBlock->getArguments()))
    state.state[arg] = newValues[i];

  return mlir::success();
};

struct BranchOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          BranchOpInterpreterInterface, mlir::cf::BranchOp> {
  mlir::FailureOr<bool> interpret(mlir::Operation *o,
                                  InterpreterState &state) const {
    auto op = mlir::cast<mlir::cf::BranchOp>(o);
    if (mlir::failed(
            jumpToBlock(op, state, op.getDest(), op.getDestOperands())))
      return mlir::failure();

    return true;
  }
};

struct CondBranchOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          CondBranchOpInterpreterInterface, mlir::cf::CondBranchOp> {
  mlir::FailureOr<bool> interpret(mlir::Operation *o,
                                  InterpreterState &state) const {
    auto op = mlir::cast<mlir::cf::CondBranchOp>(o);
    auto cond = getInt(state, op.getCondition());
    if (!cond)
      return op.emitError("Invalid cond val");

    mlir::Block *dest = (*cond ? op.getTrueDest() : op.getFalseDest());
    mlir::ValueRange destArgs =
        (*cond ? op.getTrueDestOperands() : op.getFalseDestOperands());

    if (mlir::failed(jumpToBlock(op, state, dest, destArgs)))
      return mlir::failure();

    return true;
  }
};

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

struct SelectOpDataflowJoinInterface final
    : public hc::typing::DataflowJoinInterface::ExternalModel<
          SelectOpDataflowJoinInterface, mlir::arith::SelectOp> {

  void getArgsIndices(mlir::Operation * /*op*/, unsigned resultIndex,
                      llvm::SmallVectorImpl<unsigned> &argsIndices) const {
    assert(resultIndex == 0);
    argsIndices.emplace_back(1);
    argsIndices.emplace_back(2);
  }
};
} // namespace

void hc::typing::registerArithTypingInterpreter(mlir::MLIRContext &ctx) {
  ctx.loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect>();

  mlir::cf::BranchOp::attachInterface<BranchOpInterpreterInterface>(ctx);
  mlir::cf::CondBranchOp::attachInterface<CondBranchOpInterpreterInterface>(
      ctx);

  mlir::arith::ConstantOp::attachInterface<ConstantOpInterpreterInterface>(ctx);
  mlir::arith::AddIOp::attachInterface<AddOpInterpreterInterface>(ctx);
  mlir::arith::CmpIOp::attachInterface<CmpOpInterpreterInterface>(ctx);

  mlir::arith::SelectOp::attachInterface<SelectOpDataflowJoinInterface>(ctx);
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
  return mlir::dyn_cast<mlir::Type>(it->second);
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
hc::typing::TypeResolverReturnOp::interpret(InterpreterState &state) {
  state.setCompleted();
  return true;
}

mlir::FailureOr<bool>
hc::typing::MakeIdentOp::interpret(InterpreterState &state) {
  auto name = this->getNameAttr();
  auto paramNames =
      castArrayRef<mlir::StringAttr>(this->getParamNames().getValue());
  auto paramTypes = getTypes(state, this->getParams());
  state.state[getResult()] = hc::typing::IdentType::get(
      this->getContext(), name, paramNames, paramTypes);
  return true;
}

mlir::FailureOr<bool>
hc::typing::MakeSymbolOp::interpret(InterpreterState &state) {
  auto name = this->getNameAttr();
  state.state[getResult()] =
      hc::typing::SymbolType::get(this->getContext(), name);
  return true;
}

mlir::FailureOr<bool>
hc::typing::MakeLiteralOp::interpret(InterpreterState &state) {
  state.state[getResult()] = hc::typing::LiteralType::get(getValue());
  return true;
}

mlir::FailureOr<bool>
hc::typing::GetNumArgsOp::interpret(InterpreterState &state) {
  state.state[getResult()] =
      setInt(this->getContext(), static_cast<int64_t>(state.args.size()));
  return true;
}

mlir::FailureOr<bool> hc::typing::GetArgOp::interpret(InterpreterState &state) {
  auto index = getInt(state, getIndex());
  if (!index)
    return emitOpError("Invalid index val");

  auto id = *index;
  auto args = state.args;
  if (id < 0 || id >= static_cast<decltype(id)>(args.size()))
    return emitOpError("Index out of bounds: ")
           << id << " [0, " << args.size() << "]";

  state.state[getResult()] = args[id];
  return true;
}

mlir::FailureOr<bool>
hc::typing::GetIdentNameOp::interpret(InterpreterState &state) {
  auto ident = mlir::dyn_cast_if_present<hc::typing::IdentType>(
      hc::typing::getType(state, getIdent()));
  if (!ident)
    return emitError("Invalid ident type");

  auto name = ident.getName();
  state.state[getResult()] = hc::typing::LiteralType::get(name);
  return true;
}

mlir::FailureOr<bool>
hc::typing::GetIdentParamOp::interpret(InterpreterState &state) {
  auto ident = mlir::dyn_cast_if_present<hc::typing::IdentType>(
      hc::typing::getType(state, getIdent()));
  if (!ident)
    return emitError("Invalid ident type");

  auto names = ident.getParamNames();
  auto params = ident.getParams();

  mlir::Type paramVal;
  for (auto &&[name, val] : llvm::zip_equal(names, params)) {
    if (name == getNameAttr()) {
      paramVal = val;
      break;
    }
  }
  if (!paramVal)
    return emitError("Invalid param name");

  state.state[getResult()] = paramVal;
  return true;
}

mlir::FailureOr<bool>
hc::typing::CreateSeqOp::interpret(InterpreterState &state) {
  state.state[getResult()] = SequenceType::get(getContext(), std::nullopt);
  return true;
}

mlir::FailureOr<bool>
hc::typing::AppendSeqOp::interpret(InterpreterState &state) {
  auto seq = mlir::dyn_cast_if_present<SequenceType>(
      ::hc::typing::getType(state, getSeq()));
  if (!seq)
    return emitError("Invalid seq type");

  auto arg = ::hc::typing::getType(state, getArg());
  llvm::SmallVector<mlir::Type> newArgs;
  llvm::append_range(newArgs, seq.getParams());
  newArgs.emplace_back(arg);

  state.state[getResult()] = SequenceType::get(getContext(), newArgs);
  return true;
}

mlir::FailureOr<bool> hc::typing::IsSameOp::interpret(InterpreterState &state) {
  auto lhs = hc::typing::getType(state, getLhs());
  auto rhs = hc::typing::getType(state, getRhs());
  state.state[getResult()] = setInt(getContext(), lhs == rhs);
  return true;
}

mlir::FailureOr<bool> hc::typing::CheckOp::interpret(InterpreterState &state) {
  auto val = getInt(state, getCondition());
  if (!val)
    return emitError("Inavlid condition val");
  return static_cast<bool>(*val);
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
