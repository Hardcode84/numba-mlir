// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <queue>

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

namespace hc {
#define GEN_PASS_DEF_PYTYPEINFERENCEPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

// TODO: unhardcode
static std::optional<mlir::Type> parseAnnotation(mlir::Value val) {
  auto ctx = val.getContext();
  if (auto load = val.getDefiningOp<hc::py_ir::LoadVarOp>()) {
    return hc::typing::IdentType::get(ctx, load.getName(),
                                      /*paramNames*/ std::nullopt,
                                      /*params*/ std::nullopt);
  }

  return std::nullopt;
}

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

namespace {
struct PyTypeInferencePass final
    : public hc::impl::PyTypeInferencePassBase<PyTypeInferencePass> {

  void runOnOperation() override {
    llvm::SmallDenseMap<mlir::Value, mlir::Type> typemap;

    auto rootOp = getOperation();
    rootOp->walk([&](mlir::Operation *op) {
      if (auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(op)) {
        mlir::Block &body = func.getBodyRegion().front();
        for (auto &&[arg, annotation] :
             llvm::zip_equal(body.getArguments(), func.getAnnotations())) {
          auto type = parseAnnotation(annotation);
          if (!type)
            continue;

          typemap[arg] = *type;
        }
      }
    });

    auto getType = [&](mlir::Value val) -> mlir::Type {
      auto it = typemap.find(val);
      if (it == typemap.end())
        return {};

      return it->second;
    };

    updateTypes(rootOp, getType);
  }
};
} // namespace
