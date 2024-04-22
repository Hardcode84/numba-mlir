// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"
#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

#include <queue>

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

static void updateTypes(mlir::Operation *rootOp,
                        llvm::function_ref<mlir::Type(mlir::Value)> getType) {
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

  mlir::OpBuilder builder(rootOp->getContext());
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

  void populate(mlir::Operation *rootOp) {
    rootOp->walk([&](hc::typing::TypeResolverOp op) {
      resolversMap[op.getKey()].emplace_back(op);
    });
  }

  mlir::FailureOr<bool> run(mlir::Operation *op, mlir::TypeRange types,
                            llvm::SmallVectorImpl<mlir::Type> &result) {
    mlir::Attribute key = getTypingKey(op);
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
  llvm::DenseMap<mlir::Attribute,
                 llvm::SmallVector<hc::typing::TypeResolverOp, 1>>
      resolversMap;
};

struct PyTypeInferencePass final
    : public hc::impl::PyTypeInferencePassBase<PyTypeInferencePass> {

  void runOnOperation() override {
    auto rootOp = getOperation();
    TypingInterpreter interp;
    interp.populate(rootOp);

    llvm::SmallDenseMap<mlir::Value, mlir::Type> typemap;
    auto getType = [&](mlir::Value val) -> mlir::Type {
      auto it = typemap.find(val);
      if (it == typemap.end())
        return {};

      return it->second;
    };

    llvm::SmallVector<mlir::Type> argTypes;
    llvm::SmallVector<mlir::Type> resTypes;

    auto typingVisitor = [&](hc::py_ir::PyModuleOp op) -> mlir::WalkResult {
      // TODO: Proper dataflow analysis
      auto innerVisitor = [&](mlir::Operation *innerOp) -> mlir::WalkResult {
        if (auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(innerOp)) {
          for (auto &&[arg, annotation] :
               llvm::zip_equal(func.getBlockArgs(), func.getAnnotations())) {
            auto type = getType(annotation);
            if (!type)
              continue;

            typemap[arg] = type;
          }
        }
        argTypes.clear();
        for (mlir::Value arg : innerOp->getOperands()) {
          mlir::Type type = getType(arg);
          if (!type)
            return mlir::WalkResult::advance();

          argTypes.emplace_back(type);
        }

        resTypes.clear();
        auto res = interp.run(innerOp, argTypes, resTypes);
        if (mlir::failed(res)) {
          innerOp->emitOpError("Type inference failed");
          return mlir::WalkResult::interrupt();
        }

        if (*res) {
          for (auto &&[type, res] :
               llvm::zip_equal(resTypes, innerOp->getResults())) {
            typemap[res] = type;
          }
        }

        return mlir::WalkResult::advance();
      };
      if (op->walk<mlir::WalkOrder::PreOrder>(innerVisitor).wasInterrupted())
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::skip();
    };

    if (rootOp->walk<mlir::WalkOrder::PreOrder>(typingVisitor).wasInterrupted())
      return signalPassFailure();

    updateTypes(rootOp, getType);
  }
};
} // namespace
