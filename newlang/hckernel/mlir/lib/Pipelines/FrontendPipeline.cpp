// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/FrontendPipeline.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Transforms/Passes.hpp"

static void populatePyIROptPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "PyIROptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(mlir::createCSEPass());
        p.addPass(hc::createCleanupPySetVarPass());
      }));
}

void hc::populatePyIRPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createSimplifyASTPass());
  pm.addPass(hc::createConvertPyASTToIRPass());
  populatePyIROptPasses(pm);
  pm.addPass(hc::createReconstuctPySSAPass());
  populatePyIROptPasses(pm);
  pm.addPass(mlir::createCanonicalizerPass());
}

void hc::populateFrontendPipeline(mlir::PassManager &pm) {
  populatePyIRPipeline(pm);
  pm.addPass(hc::createPyTypeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void hc::populateDecoratorPipeline(mlir::PassManager &pm,
                                   const OpConstructorMap &opConstr) {
  pm.addPass(hc::createConvertLoadvarTypingPass(opConstr));
  pm.addPass(hc::createInlineForceInlinedFuncsPass());
}
