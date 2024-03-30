// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "mlir/Pass/PassManager.h"
#include <mlir/Pass/Pass.h>

namespace {
struct CompositePass final
    : public mlir::PassWrapper<CompositePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompositePass)

  CompositePass(std::string name_,
                llvm::function_ref<void(mlir::OpPassManager &)> populateFunc,
                unsigned maxIterations)
      : name(std::move(name_)),
        dynamicPM(std::make_shared<mlir::OpPassManager>()),
        maxIters(maxIterations) {
    populateFunc(*dynamicPM);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    dynamicPM->getDependentDialects(registry);
  }

  void runOnOperation() override {
    auto op = getOperation();
    mlir::OperationFingerPrint fp(op);

    unsigned currentIter = 0;
    while (true) {
      if (mlir::failed(runPipeline(*dynamicPM, op)))
        return signalPassFailure();

      if (currentIter++ >= maxIters) {
        op->emitWarning("Composite pass \"" + llvm::Twine(name) +
                        "\"+ didn't converge in " + llvm::Twine(maxIters) +
                        " iterations");
        break;
      }

      mlir::OperationFingerPrint newFp(op);
      if (newFp == fp)
        break;

      fp = newFp;
    }
  }

protected:
  llvm::StringRef getName() const override {
    assert(!name.empty());
    return name;
  }

private:
  std::string name;
  std::shared_ptr<mlir::OpPassManager> dynamicPM;
  unsigned maxIters;
};
} // namespace

std::unique_ptr<mlir::Pass> hc::createCompositePass(
    std::string name,
    llvm::function_ref<void(mlir::OpPassManager &)> populateFunc,
    unsigned maxIterations) {
  assert(!name.empty());
  assert(populateFunc);
  return std::make_unique<CompositePass>(std::move(name), populateFunc,
                                         maxIterations);
}
