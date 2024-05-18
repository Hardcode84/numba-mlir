// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

namespace mlir {
class MLIRContext;
}

bool compileAST(mlir::MLIRContext &ctx, const std::string &source,
                const std::string &funcName);
