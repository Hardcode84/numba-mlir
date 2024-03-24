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

void populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns);
void populateConvertPyASTToIRPatterns(mlir::RewritePatternSet &patterns);
} // namespace hc
