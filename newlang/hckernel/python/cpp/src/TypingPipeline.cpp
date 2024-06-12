// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingPipeline.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Pipelines/FrontendPipeline.hpp"

static std::optional<mlir::ArrayAttr>
processDecorators(mlir::ValueRange decorators) {
  if (decorators.size() != 1)
    return std::nullopt;

  auto type =
      mlir::dyn_cast<hc::typing::IdentType>(decorators.front().getType());
  if (!type || type.getName() != "type_resolver_type" ||
      type.getParamNames().size() != 1 || type.getParamNames().front() != "key")
    return std::nullopt;

  auto seqType =
      mlir::dyn_cast<hc::typing::SequenceType>(type.getParams().front());
  if (!seqType)
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute> ret;
  for (auto elem : seqType.getParams()) {
    auto lit = mlir::dyn_cast<hc::typing::LiteralType>(elem);
    if (!lit)
      return std::nullopt;

    ret.emplace_back(lit.getValue());
  }
  return mlir::ArrayAttr::get(type.getContext(), ret);
}

namespace {
struct GenResolversPass
    : public mlir::PassWrapper<GenResolversPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenResolversPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());
    auto visitor = [&](hc::py_ir::PyModuleOp mod) -> mlir::WalkResult {
      builder.setInsertionPoint(mod);
      for (auto func : mod.getBody()->getOps<hc::py_ir::PyFuncOp>()) {
        if (!func.getCaptureArgs().empty()) {
          func->emitOpError("Cannot convert function with captures");
          return mlir::WalkResult::interrupt();
        }

        if (!llvm::all_of(func.getBlockArgs(), [](auto arg) {
              return mlir::isa<hc::typing::ValueType>(arg.getType());
            })) {
          func->emitOpError("Invalid arg types");
          return mlir::WalkResult::interrupt();
        }

        auto decorator = processDecorators(func.getDecorators());
        if (!decorator) {
          func->emitOpError("Invalid decorator");
          return mlir::WalkResult::interrupt();
        }

        mlir::Location loc = func.getLoc();
        auto resolver =
            builder.create<hc::typing::TypeResolverOp>(loc, *decorator);
        mlir::Region &dstReg = resolver.getBodyRegion();
        builder.inlineRegionBefore(func.getBodyRegion(), dstReg,
                                   dstReg.begin());

        mlir::Block &entry = dstReg.front();
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&entry);
        for (auto &&[i, arg] : llvm::enumerate(entry.getArguments())) {
          mlir::Value idx =
              builder.create<mlir::arith::ConstantIndexOp>(loc, i);
          mlir::Value newArg =
              builder.create<hc::typing::GetArgOp>(loc, arg.getType(), idx);
          arg.replaceAllUsesWith(newArg);
        }
        entry.eraseArguments(0, entry.getNumArguments());

        for (mlir::Block &block : dstReg) {
          auto term =
              mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
          if (!term)
            continue;

          builder.setInsertionPoint(term);
          builder.replaceOpWithNewOp<hc::typing::TypeResolverReturnOp>(
              term, term.getOperand());
        }
      }

      mod->erase();
      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PreOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

void populateTypingPipeline(mlir::PassManager &pm) {
  hc::populateFrontendPipeline(pm);
  pm.addPass(std::make_unique<GenResolversPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
