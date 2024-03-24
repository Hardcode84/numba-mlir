// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_CONVERTPYASTTOIRPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static mlir::Value getVar(mlir::OpBuilder & /*builder*/, mlir::Location /*loc*/,
                          mlir::Value val) {
  // TODO
  return val;
}

namespace {
class ConvertModule final
    : public mlir::OpRewritePattern<hc::py_ast::PyModuleOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::PyModuleOp op,
                  mlir::PatternRewriter &rewriter) const override {
    {
      auto term =
          mlir::cast<hc::py_ast::PyModuleEndOp>(op.getBody(0)->getTerminator());
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<hc::py_ir::PyModuleEndOp>(term);
    }

    auto newMod = rewriter.create<hc::py_ir::PyModuleOp>(op.getLoc());

    mlir::Region &dstRegion = newMod.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static bool checkFuncReturn(mlir::Block *body) {
  auto termIt = body->getTerminator()->getIterator();
  if (body->begin() == termIt)
    return false;

  auto retOp = mlir::dyn_cast<hc::py_ast::ReturnOp>(*std::prev(termIt));
  return retOp && retOp.getValue();
}

class ConvertFunc final : public mlir::OpRewritePattern<hc::py_ast::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!checkFuncReturn(op.getBody()))
      return mlir::failure();

    // TODO
    llvm::SmallVector<mlir::Type> argTypes;
    llvm::SmallVector<mlir::Value> decorators;

    mlir::Location loc = op.getLoc();
    auto newOp = rewriter.create<hc::py_ir::PyFuncOp>(loc, op.getName(),
                                                      argTypes, decorators);
    mlir::Region &dstRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    mlir::Block &entryBlock = dstRegion.front();
    mlir::Block &bodyBlock = dstRegion.back();

    mlir::OpBuilder::InsertionGuard g(rewriter);

    rewriter.setInsertionPointToEnd(&entryBlock);
    rewriter.create<mlir::cf::BranchOp>(loc, &bodyBlock);
    rewriter.eraseOp(op);
    llvm::errs() << "asdasd 5\n";
    return mlir::success();
  }
};

class ConvertReturn final
    : public mlir::OpRewritePattern<hc::py_ast::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!mlir::isa<hc::py_ir::PyFuncOp>(op->getParentOp()))
      return mlir::failure();

    auto term = op->getBlock()->getTerminator();
    if (&*std::next(op->getIterator()) != term)
      return mlir::failure();

    mlir::Value val = op.getValue();
    if (!val)
      return mlir::failure();

    val = getVar(rewriter, op.getLoc(), val);
    rewriter.replaceOpWithNewOp<hc::py_ir::ReturnOp>(op, val);
    rewriter.eraseOp(term);
    return mlir::success();
  }
};

struct ConvertPyASTToIRPass final
    : public hc::impl::ConvertPyASTToIRPassBase<ConvertPyASTToIRPass> {

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    hc::populateConvertPyASTToIRPatterns(patterns);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void hc::populateConvertPyASTToIRPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertModule, ConvertFunc, ConvertReturn>(
      patterns.getContext());
}
