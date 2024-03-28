// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir::cf {
class ControlFlowDialect;
}

namespace hc {
#define GEN_PASS_DECL
#include "hc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hc/Transforms/Passes.h.inc"

/// Create composite pass, which runs selected set of passes until fixed point
/// or maximum number of iterations reached.
std::unique_ptr<mlir::Pass> createCompositePass(
    std::string name, std::function<void(mlir::OpPassManager &)> populateFunc,
    unsigned maxIterations = std::numeric_limits<unsigned>::max());

void populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns);
void populateConvertPyASTToIRPatterns(mlir::RewritePatternSet &patterns);
} // namespace hc
