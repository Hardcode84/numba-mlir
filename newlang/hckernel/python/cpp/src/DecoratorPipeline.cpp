// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingDispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include "CompilerFront.hpp"
#include "Context.hpp"
#include "MlirWrappers.hpp"
#include "hc/Pipelines/FrontendPipeline.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include "DecoratorPipeline.hpp"
#include "hc/Transforms/Passes.hpp"

namespace py = pybind11;

[[noreturn]] static void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}

DecoratorPipeline::DecoratorPipeline(Context &context) : context(context) {}

void DecoratorPipeline::addVariable(std::string name, MlirOpWrapper *wrapper) {
  variables[name] = wrapper->getOpConstrutor();
}

void DecoratorPipeline::addDecorator(std::string name,
                                     MlirWrapperBase *wrapper) {
  decorators[name] = wrapper;
}

mlir::Operation *DecoratorPipeline::run(mlir::Operation *mod) {
  auto *mlirContext = &context.context;
  mlir::PassManager pm(mlirContext);

  pm.addPass(hc::createConvertLoadvarTypingPass(this->variables));
  pm.addPass(hc::createInlineForceInlinedFuncsPass());

  if (context.settings.dumpIR) {
    pm.enableIRPrinting();
  }

  runUnderDiag(pm, mod);

  return mod;
}
