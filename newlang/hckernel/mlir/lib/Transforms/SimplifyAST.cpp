// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_SIMPLIFYASTPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
class ReturnOpNoArg final
    : public mlir::OpRewritePattern<hc::py_ast::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getValue())
      return mlir::failure();

    mlir::Attribute attr = hc::py_ast::NoneAttr::get(getContext());
    mlir::Value arg =
        rewriter.create<hc::py_ast::ConstantOp>(op.getLoc(), attr);
    rewriter.replaceOpWithNewOp<hc::py_ast::ReturnOp>(op, arg);
    return mlir::success();
  }
};

class CleanupPassOp final : public mlir::OpRewritePattern<hc::py_ast::PassOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::PassOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CleanupNoReturn final
    : public mlir::OpTraitRewritePattern<hc::py_ast::NoReturn> {
public:
  using OpTraitRewritePattern::OpTraitRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block *block = op->getBlock();
    mlir::Operation *term = block->getTerminator();
    if (term == op)
      return mlir::failure();

    auto begin = op->getIterator();
    auto it = std::prev(term->getIterator());
    bool changed = false;
    while (it != begin) {
      mlir::Operation &current = *it;
      it = std::prev(it);
      if (current.use_empty()) {
        rewriter.eraseOp(&current);
        changed = true;
      }
    }
    return mlir::success(changed);
  }
};

struct SimplifyASTPass final
    : public hc::impl::SimplifyASTPassBase<SimplifyASTPass> {

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    hc::populateSimplifyASTPatterns(patterns);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void hc::populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ReturnOpNoArg, CleanupPassOp, CleanupNoReturn>(
      patterns.getContext());
}
