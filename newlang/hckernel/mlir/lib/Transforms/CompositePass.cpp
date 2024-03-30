// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "mlir/Pass/PassManager.h"
#include <mlir/Pass/Pass.h>

namespace {
struct CompositePass final
    : public mlir::PassWrapper<CompositePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompositePass)

  CompositePass(std::string name_,
                std::function<void(mlir::OpPassManager &)> func,
                unsigned maxIterations)
      : name(std::move(name_)), populateFunc(std::move(func)),
        dynamicPM(std::make_shared<mlir::OpPassManager>()),
        maxIters(maxIterations) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    populateFunc(*dynamicPM);
    dynamicPM->getDependentDialects(registry);
  }

  void runOnOperation() override {
    assert(populateFunc);
    auto op = getOperation();
    mlir::OperationFingerPrint fp(op);

    unsigned currentIter = 0;
    populateFunc(*dynamicPM);
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
  std::function<void(mlir::OpPassManager &)> populateFunc;
  std::shared_ptr<mlir::OpPassManager> dynamicPM;
  unsigned maxIters;
};
} // namespace

std::unique_ptr<mlir::Pass>
hc::createCompositePass(std::string name,
                        std::function<void(mlir::OpPassManager &)> populateFunc,
                        unsigned maxIterations) {
  assert(!name.empty());
  assert(populateFunc);
  return std::make_unique<CompositePass>(
      std::move(name), std::move(populateFunc), maxIterations);
}
