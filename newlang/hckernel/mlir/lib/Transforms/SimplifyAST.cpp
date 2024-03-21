// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hckernel/Transforms/Passes.hpp"

#include "hckernel/Dialect/PyAST/IR/PyASTOps.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hckernel {
#define GEN_PASS_DEF_SIMPLIFYASTPASS
#include "hckernel/Transforms/Passes.h.inc"
} // namespace hckernel

namespace {
class CleanupPassOp final
    : public mlir::OpRewritePattern<hckernel::py_ast::PassOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hckernel::py_ast::PassOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CleanupNoReturn final
    : public mlir::OpTraitRewritePattern<hckernel::py_ast::NoReturn> {
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
    : public hckernel::impl::SimplifyASTPassBase<SimplifyASTPass> {

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    hckernel::populateSimplifyASTPatterns(patterns);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void hckernel::populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<CleanupPassOp, CleanupNoReturn>(patterns.getContext());
}
