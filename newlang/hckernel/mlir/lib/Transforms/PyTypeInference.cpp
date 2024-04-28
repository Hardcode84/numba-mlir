// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"
#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

#include <queue>

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

namespace hc {
#define GEN_PASS_DEF_PYTYPEINFERENCEPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static mlir::Value makeCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val, mlir::Type newType) {
  if (val.getType() == newType)
    return val;

  return builder.create<hc::py_ir::CastOp>(loc, newType, val);
}

static unsigned getSuccessorIndex(mlir::Operation *op, mlir::Block *successor) {
  for (auto &&[i, s] : llvm::enumerate(op->getSuccessors())) {
    if (s == successor)
      return static_cast<unsigned>(i);
  }
  llvm_unreachable("Invalid successor");
}

static void
updatePredecessorTypes(mlir::OpBuilder &builder, mlir::Block *block,
                       llvm::function_ref<mlir::Type(mlir::Value)> getType) {
  auto getTypeChecked = [&](mlir::Value arg) -> mlir::Type {
    if (auto t = getType(arg))
      return t;

    return arg.getType();
  };
  mlir::ValueRange blockArgs = block->getArguments();
  for (mlir::Block *predecessor : block->getPredecessors()) {
    auto term =
        mlir::cast<mlir::BranchOpInterface>(predecessor->getTerminator());
    auto idx = getSuccessorIndex(term, block);
    mlir::ValueRange successorArgs =
        term.getSuccessorOperands(idx).getForwardedOperands();
    mlir::Location loc = term.getLoc();
    builder.setInsertionPoint(term);
    for (auto &&[srcArg, dstArg] : llvm::zip_equal(successorArgs, blockArgs)) {
      mlir::Type srcType = getTypeChecked(srcArg);
      mlir::Type dstType = getTypeChecked(dstArg);
      if (srcType == dstType)
        continue;

      mlir::Value cast = makeCast(builder, loc, srcArg, dstType);
      auto shouldReplace = [&](mlir::OpOperand &op) -> bool {
        return op.getOwner() == term;
      };

      srcArg.replaceUsesWithIf(cast, shouldReplace);
    }
  }
}

static bool canUpdateValType(mlir::Value val) {
  auto def = val.getDefiningOp();

  if (!def)
    def = val.getParentBlock()->getParentOp();

  auto iface =
      mlir::dyn_cast_if_present<hc::typing::TypingUpdateInplaceInterface>(def);
  return iface && iface.canUpdateArgTypeInplace(val);
}

static bool UpdateInplace = true;

static void updateTypes(mlir::Operation *rootOp,
                        llvm::function_ref<mlir::Type(mlir::Value)> getType) {
  mlir::OpBuilder builder(rootOp->getContext());
  if (UpdateInplace) {
    auto needWrap = [&](mlir::Operation *op) -> bool {
      if (op->mightHaveTrait<mlir::OpTrait::IsTerminator>())
        return false;

      auto iface = mlir::dyn_cast<hc::typing::TypingUpdateInplaceInterface>(op);
      for (auto args : {mlir::ValueRange(op->getOperands()),
                        mlir::ValueRange(op->getResults())}) {
        for (mlir::Value arg : args) {
          mlir::Type oldType = arg.getType();
          mlir::Type newType = getType(arg);
          if (!newType || newType == oldType)
            continue;

          if (!iface || !iface.canUpdateArgTypeInplace(arg))
            return true;
        }
      }
      return false;
    };
    rootOp->walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *op) {
      if (!needWrap(op))
        return;

      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();
      auto resolve = builder.create<hc::typing::ResolveOp>(
          loc, op->getResultTypes(), op->getOperands());
      mlir::Block *body = resolve.getBody();
      assert(body->getNumArguments() == op->getNumOperands());
      op->setOperands(body->getArguments());
      op->replaceAllUsesWith(resolve.getResults());

      op->moveBefore(body, body->begin());
      builder.setInsertionPointToEnd(body);
      builder.create<hc::typing::ResolveYieldOp>(loc, op->getResults());
    });
    auto updateTypes = [&](mlir::ValueRange resVals,
                           mlir::ValueRange typeVals) {
      for (auto &&[resArg, typeArg] : llvm::zip_equal(resVals, typeVals)) {
        mlir::Type oldType = typeArg.getType();
        mlir::Type newType = getType(typeArg);
        if (!newType || newType == oldType)
          continue;

        if (canUpdateValType(resArg)) {
          resArg.setType(newType);
        } else {
          builder.setInsertionPointAfterValue(resArg);
          mlir::Value cast =
              makeCast(builder, resArg.getLoc(), resArg, newType);
          resArg.replaceAllUsesExcept(cast, cast.getDefiningOp());
        }
      }
    };
    rootOp->walk([&](mlir::Block *block) {
      updateTypes(block->getArguments(), block->getArguments());
      updatePredecessorTypes(builder, block, getType);
    });
    rootOp->walk([&](mlir::Operation *op) {
      if (auto resolve = mlir::dyn_cast<hc::typing::ResolveOp>(op)) {
        mlir::Operation &innerOp = resolve.getBody()->front();
        return updateTypes(op->getResults(), innerOp.getResults());
      }
      if (mlir::isa_and_present<hc::typing::ResolveOp>(op->getParentOp()))
        return;

      updateTypes(op->getResults(), op->getResults());
    });
    return;
  }

  llvm::SetVector<mlir::Operation *> opsToUpdate;
  rootOp->walk([&](mlir::Operation *op) {
    if (op->mightHaveTrait<mlir::OpTrait::IsTerminator>())
      return;

    auto typeChanged = [&](mlir::Value val) -> bool {
      auto type = getType(val);
      return type && type != val.getType();
    };
    if (llvm::any_of(op->getOperands(), typeChanged) ||
        llvm::any_of(op->getResults(), typeChanged))
      opsToUpdate.insert(op);
  });

  for (mlir::Operation *op : opsToUpdate) {
    builder.setInsertionPoint(op);
    mlir::Location loc = op->getLoc();
    auto resolve = builder.create<hc::typing::ResolveOp>(
        loc, op->getResultTypes(), op->getOperands());
    mlir::Block *body = resolve.getBody();
    assert(body->getNumArguments() == op->getNumOperands());
    op->setOperands(body->getArguments());
    op->replaceAllUsesWith(resolve.getResults());

    op->moveBefore(body, body->begin());
    builder.setInsertionPointToEnd(body);
    builder.create<hc::typing::ResolveYieldOp>(loc, op->getResults());
  }

  auto updateUses = [&](mlir::Value val, mlir::Type oldType) {
    mlir::Value casted;
    for (mlir::OpOperand &use : llvm::make_early_inc_range(val.getUses())) {
      mlir::Operation *owner = use.getOwner();
      if (!owner->mightHaveTrait<mlir::OpTrait::IsTerminator>())
        continue;

      if (!casted)
        casted = makeCast(builder, val.getLoc(), val, oldType);

      use.set(casted);
    }
  };

  for (mlir::Operation *op : opsToUpdate) {
    auto resolve = op->getParentOfType<hc::typing::ResolveOp>();
    assert(resolve);
    builder.setInsertionPointAfter(resolve);

    for (auto &&[origVal, newVal] :
         llvm::zip_equal(op->getResults(), resolve->getResults())) {
      mlir::Type oldType = origVal.getType();
      mlir::Type newType = getType(origVal);
      if (!newType || oldType == newType)
        continue;

      newVal.setType(newType);

      updateUses(newVal, oldType);
    }
  }

  rootOp->walk([&](mlir::Block *block) {
    builder.setInsertionPointToStart(block);
    for (mlir::Value arg : block->getArguments()) {
      mlir::Type oldType = arg.getType();
      mlir::Type newType = getType(arg);
      if (!newType || oldType == newType)
        continue;

      mlir::Value newArg = makeCast(builder, arg.getLoc(), arg, newType);
      arg.replaceAllUsesExcept(newArg, newArg.getDefiningOp());

      updateUses(newArg, oldType);
    }
  });
}

static mlir::Attribute getTypingKey(mlir::Operation *op) {
  if (auto iface = mlir::dyn_cast<hc::typing::TypingKeyInterface>(op))
    return iface.getTypingKey();

  return nullptr;
}

namespace {
struct TypingInterpreter {
  TypingInterpreter(mlir::MLIRContext *ctx) {
    joinTypesKey = mlir::StringAttr::get(ctx, "join_types");
  }

  void populate(mlir::Operation *rootOp) {
    rootOp->walk([&](hc::typing::TypeResolverOp op) {
      resolversMap[op.getKey()].emplace_back(op);
    });
  }

  mlir::FailureOr<bool> run(mlir::Operation *op, mlir::TypeRange types,
                            llvm::SmallVectorImpl<mlir::Type> &result) {
    if (auto iface = mlir::dyn_cast<hc::typing::TypeInferenceInterface>(op)) {
      auto res = iface.inferTypes(types, result);
      if (mlir::failed(res) || *res)
        return res;
    }

    mlir::Attribute key = getTypingKey(op);
    return run(key, types, result);
  }

  mlir::FailureOr<bool>
  runJoinTypes(mlir::TypeRange types,
               llvm::SmallVectorImpl<mlir::Type> &result) {
    return run(joinTypesKey, types, result);
  }

  mlir::FailureOr<bool> run(mlir::Attribute key, mlir::TypeRange types,
                            llvm::SmallVectorImpl<mlir::Type> &result) {
    if (!key)
      return false;

    auto it = resolversMap.find(key);
    if (it == resolversMap.end())
      return false;

    for (auto resolverOp : it->second) {
      auto res = interp.run(resolverOp, types, result);
      if (mlir::failed(res))
        return mlir::failure();

      if (*res)
        return true;
    }
    return false;
  }

private:
  hc::typing::Interpreter interp;
  mlir::Attribute joinTypesKey;
  llvm::DenseMap<mlir::Attribute,
                 llvm::SmallVector<hc::typing::TypeResolverOp, 1>>
      resolversMap;
};

struct TypeValue {
  TypeValue() = default;
  TypeValue(TypingInterpreter *interp, mlir::Type t = {})
      : interpreter(interp), type(t) {}
  TypeValue(std::string desc) : errDesc(std::move(desc)) {
    assert(!errDesc.empty());
  }

  static TypeValue getInvalid(mlir::Operation *op) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "Type inference failed for op: ";
    os << *op;
    os.flush();

    return TypeValue(str);
  }

  static TypeValue getInvalid(const TypeValue &lhs, const TypeValue &rhs) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "Incompatible types: ";
    os << lhs.type << " and " << rhs.type;
    os.flush();

    return TypeValue(str);
  }

  static TypeValue join(const TypeValue &lhs, const TypeValue &rhs) {
    if (!lhs.isValid())
      return lhs;

    if (!rhs.isValid())
      return rhs;

    if (!lhs.isInitialized())
      return rhs;

    if (!rhs.isInitialized())
      return lhs;

    mlir::Type lhsType = lhs.type;
    mlir::Type rhsType = rhs.type;
    if (lhsType == rhsType)
      return lhs;

    if (!lhsType || mlir::isa<hc::py_ir::UndefinedType>(lhsType))
      return rhs;

    if (!rhsType || mlir::isa<hc::py_ir::UndefinedType>(rhsType))
      return lhs;

    assert(lhs.interpreter == rhs.interpreter);
    auto interp = lhs.interpreter;
    mlir::SmallVector<mlir::Type> res;
    if (mlir::failed(interp->runJoinTypes({lhsType, rhsType}, res)) ||
        res.size() != 1)
      return getInvalid(lhs, rhs);

    return TypeValue(interp, res.front());
  }

  bool operator==(const TypeValue &rhs) const { return type == rhs.type; }

  void print(llvm::raw_ostream &os) const {
    if (!isValid()) {
      os << "Invalid: " << errDesc;
    } else if (!isInitialized()) {
      os << "Uninitialized";
    } else {
      os << type;
    }
  }

  mlir::Type getType() const { return type; }

  bool isInitialized() const { return isValid() && type != nullptr; }

  bool isValid() const { return errDesc.empty(); }

  mlir::StringRef getErrDesc() const {
    assert(!isValid());
    return errDesc;
  }

private:
  TypingInterpreter *interpreter = nullptr;
  mlir::Type type;
  std::string errDesc;
};

struct TypeValueLattice : public mlir::dataflow::Lattice<TypeValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeValueLattice)
  using Lattice::Lattice;
};

class TypeAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<TypeValueLattice> {
public:
  TypeAnalysis(mlir::DataFlowSolver &solver, TypingInterpreter &interp)
      : SparseForwardDataFlowAnalysis(solver), interpreter(interp) {}

  void visitOperation(mlir::Operation *op,
                      llvm::ArrayRef<const TypeValueLattice *> operands,
                      llvm::ArrayRef<TypeValueLattice *> results) override {
    llvm::SmallVector<mlir::Type> argTypes;
    for (auto arg : operands) {
      auto &latticeVal = arg->getValue();
      if (!latticeVal.isInitialized() || !latticeVal.isValid())
        return;

      argTypes.emplace_back(latticeVal.getType());
    }

    llvm::SmallVector<mlir::Type> result;
    auto res = interpreter.run(op, argTypes, result);
    if (mlir::failed(res)) {
      auto errState = TypeValue::getInvalid(op);
      for (auto resultLattice : results) {
        auto changed = resultLattice->join(errState);
        propagateIfChanged(resultLattice, changed);
      }
      return;
    }

    if (!*res)
      return;

    assert(result.size() == results.size());
    for (auto &&[resultLattice, res] : llvm::zip_equal(results, result)) {
      auto changed = resultLattice->join(TypeValue(&interpreter, res));
      propagateIfChanged(resultLattice, changed);
    }
  }

  void
  visitNonControlFlowArguments(mlir::Operation *op,
                               const mlir::RegionSuccessor &successor,
                               mlir::ArrayRef<TypeValueLattice *> argLattices,
                               unsigned firstIndex) override {
    if (auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(op)) {
      for (auto &&[annotation, arg] :
           llvm::zip_equal(func.getAnnotations(), func.getBlockArgs())) {
        TypeValueLattice *annotationLattice = getLatticeElement(annotation);
        TypeValue val = annotationLattice->getValue();
        TypeValueLattice *argLattice = getLatticeElement(arg);
        propagateIfChanged(argLattice, argLattice->join(val));
      }
      for (auto &&[capture, arg] :
           llvm::zip_equal(func.getCaptureArgs(), func.getCaptureBlockArgs())) {
        TypeValueLattice *annotationLattice = getLatticeElement(capture);
        TypeValue val = annotationLattice->getValue();
        TypeValueLattice *argLattice = getLatticeElement(arg);
        propagateIfChanged(argLattice, argLattice->join(val));
      }
    }

    return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, argLattices, firstIndex);
  }

  void setToEntryState(TypeValueLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(TypeValue(&interpreter)));
  }

private:
  TypingInterpreter &interpreter;
};

static mlir::Operation *getOp(mlir::Value val) {
  if (auto defOp = val.getDefiningOp())
    return defOp;

  return val.getParentRegion()->getParentOp();
}

struct PyTypeInferencePass final
    : public hc::impl::PyTypeInferencePassBase<PyTypeInferencePass> {

  void runOnOperation() override {
    auto rootOp = getOperation();
    TypingInterpreter interp(&getContext());
    interp.populate(rootOp);

    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<TypeAnalysis>(interp);
    if (mlir::failed(solver.initializeAndRun(rootOp)))
      return signalPassFailure();

    auto getType = [&](mlir::Value arg) -> mlir::Type {
      auto *state = solver.lookupState<TypeValueLattice>(arg);
      if (!state)
        return {};

      auto &val = state->getValue();
      if (!val.isValid()) {
        getOp(arg)->emitError(val.getErrDesc());
        signalPassFailure();
        return {};
      }
      if (!val.isInitialized())
        return {};

      return val.getType();
    };

    updateTypes(rootOp, getType);
  }
};
} // namespace
