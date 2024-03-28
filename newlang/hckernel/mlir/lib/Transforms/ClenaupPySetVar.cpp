// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

namespace hc {
#define GEN_PASS_DEF_CLEANUPPYSETVARPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct CleanupPySetVarPass final
    : public hc::impl::CleanupPySetVarPassBase<CleanupPySetVarPass> {

  void runOnOperation() override {
    llvm::SmallDenseSet<mlir::StringAttr> liveNames;
    getOperation()->walk([&](hc::py_ir::PyModuleOp mod) {
      liveNames.clear();
      mod.walk([&](hc::py_ir::LoadVarOp load) {
        liveNames.insert(load.getNameAttr());
      });

      mod.walk([&](hc::py_ir::StoreVarOp store) {
        if (!liveNames.contains(store.getNameAttr()))
          store->erase();
      });
    });
  }
};
} // namespace
