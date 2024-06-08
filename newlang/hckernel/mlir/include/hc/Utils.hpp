// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <utility>

#include <functional>
#include <llvm/ADT/ScopeExit.h>
#include <string>
#include <unordered_map>

namespace mlir {
class PatternRewriter;
class Operation;
} // namespace mlir

namespace hc {
template <typename T, typename H, typename F>
inline auto scopedDiagHandler(T &ctx, H &&diag_handler, F &&func) {
  auto &diagEngine = ctx.getDiagEngine();
  auto diagId = diagEngine.registerHandler(std::forward<H>(diag_handler));
  auto diagGuard =
      llvm::make_scope_exit([&]() { diagEngine.eraseHandler(diagId); });
  return func();
}

using OpConstructorFunc =
    std::function<mlir::Operation *(mlir::PatternRewriter &)>;
using OpConstructorMap = std::unordered_map<std::string, OpConstructorFunc>;
} // namespace hc
