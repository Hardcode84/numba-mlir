// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingPipeline.hpp"

#include "hc/Pipelines/FrontendPipeline.hpp"

void populateTypingPipeline(mlir::PassManager &pm) {
  hc::populateFrontendPipeline(pm);
}
