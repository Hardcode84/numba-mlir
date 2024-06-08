// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hc/Utils.hpp"

namespace mlir {
class PassManager;
}

namespace hc {
void populatePyIRPipeline(mlir::PassManager &pm);
void populateFrontendPipeline(mlir::PassManager &pm);
void populateDecoratorPipeline(mlir::PassManager &pm,
                               const OpConstructorMap &opConstr);
} // namespace hc
