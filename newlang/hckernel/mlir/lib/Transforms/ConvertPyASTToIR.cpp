// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_CONVERTPYASTTOIRPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

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
  patterns.insert<ConvertModule>(patterns.getContext());
}
