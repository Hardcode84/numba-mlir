// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hc/Utils.hpp"
#include <mlir/Pass/Pass.h>

namespace mlir {
class Pass;
namespace cf {
class ControlFlowDialect;
}
} // namespace mlir

namespace hc {
#define GEN_PASS_DECL
#include "hc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hc/Transforms/Passes.h.inc"

void populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns);
void populateConvertPyASTToIRPatterns(mlir::RewritePatternSet &patterns);
void populateConvertLoadvarTypingPatterns(
    mlir::RewritePatternSet &patterns, const hc::OpConstructorMap &opConstrMap);
void populateConvertFuncTypingPatterns(mlir::RewritePatternSet &patterns);
void populateInlineForceInlinedPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass>
createConvertLoadvarTypingPass(const hc::OpConstructorMap &opConstrMap);
} // namespace hc
