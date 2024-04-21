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

static mlir::Value skipCasts(mlir::Value val) {
  while (auto cast = val.getDefiningOp<hc::py_ir::CastOp>())
    val = cast.getValue();

  return val;
}

// TODO: unhardcode
static std::optional<mlir::Type> matchSymbolOrLiteral(mlir::Value val) {
  val = skipCasts(val);
  if (auto c = val.getDefiningOp<hc::py_ir::ConstantOp>())
    return hc::typing::LiteralType::get(c.getValue());

  if (auto load = val.getDefiningOp<hc::py_ir::LoadVarOp>())
    return hc::typing::SymbolType::get(val.getContext(), load.getName());

  return std::nullopt;
}

static std::optional<mlir::Type> matchBuffer(mlir::Value val) {
  auto getitem = val.getDefiningOp<hc::py_ir::GetItemOp>();
  if (!getitem)
    return std::nullopt;

  auto buff = getitem.getTarget().getDefiningOp<hc::py_ir::LoadVarOp>();
  if (!buff || buff.getName() != "Buffer")
    return std::nullopt;

  auto shape = getitem.getIndex().getDefiningOp<hc::py_ir::TuplePackOp>();
  if (!shape)
    return std::nullopt;

  llvm::SmallVector<mlir::Type> shapeTypes;
  for (auto arg : shape.getArgs()) {
    auto type = matchSymbolOrLiteral(arg);
    if (!type)
      return std::nullopt;

    shapeTypes.emplace_back(*type);
  }
  auto ctx = val.getContext();
  mlir::Type seq = hc::typing::SequenceType::get(ctx, shapeTypes);
  return hc::typing::IdentType::get(ctx, "Buffer", mlir::StringRef("Shape"),
                                    seq);
}

static std::optional<mlir::Type> parseAnnotation(mlir::Value val) {
  if (auto buffer = matchBuffer(val))
    return buffer;

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

  mlir::FailureOr<llvm::SmallVector<mlir::Type>> run(mlir::Operation *op,
                                                     mlir::TypeRange types) {
    mlir::Attribute key = getTypingKey(op);
    if (!key)
      return mlir::failure();

    auto it = resolversMap.find(key);
    if (it == resolversMap.end())
      return mlir::failure();

    for (auto resolverOp : it->second) {
      auto res = interp.run(resolverOp, types);
      if (mlir::succeeded(res))
        return res;
    }
    return mlir::failure();
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

    auto typingVisitor = [&](hc::py_ir::PyModuleOp op) -> mlir::WalkResult {
      // TODO: Proper dataflow analysis
      op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *innerOp) {
        if (auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(innerOp)) {
          for (auto &&[arg, annotation] :
               llvm::zip_equal(func.getBlockArgs(), func.getAnnotations())) {
            auto type = parseAnnotation(annotation);
            if (!type)
              continue;

            typemap[arg] = *type;
          }
        }
        argTypes.clear();
        for (mlir::Value arg : innerOp->getOperands()) {
          mlir::Type type = getType(arg);
          if (!type)
            return;

          argTypes.emplace_back(type);
        }

        auto resTypes = interp.run(innerOp, argTypes);
        if (mlir::failed(resTypes))
          return;

        for (auto &&[type, res] :
             llvm::zip_equal(*resTypes, innerOp->getResults()))
          typemap[res] = type;
      });
      return mlir::WalkResult::skip();
    };

    rootOp->walk<mlir::WalkOrder::PreOrder>(typingVisitor);

    updateTypes(rootOp, getType);
  }
};
} // namespace
