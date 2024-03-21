// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hckernel/Transforms/Passes.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hckernel {
#define GEN_PASS_DEF_SIMPLIFYASTPASS
#include "hckernel/Transforms/Passes.h.inc"
} // namespace hckernel

namespace {
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

void hckernel::populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns) {}
