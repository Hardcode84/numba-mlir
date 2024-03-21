// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

namespace hckernel {
#define GEN_PASS_DECL
#include "hckernel/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hckernel/Transforms/Passes.h.inc"

void populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns);
} // namespace hckernel
