// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>

#include "MlirWrappers.hpp"
#include <pybind11/pybind11.h>

#include "unordered_map"

namespace mlir {
class MLIRContext;
}

struct Context;

class DecoratorPipeline {
public:
  DecoratorPipeline(Context &context);

  void addVariable(std::string name, MlirOpWrapper *wrapper);
  void addDecorator(std::string name, MlirWrapperBase *wrapper);

  mlir::Operation *run(mlir::Operation *mod);

private:
  Context &context;

  hc::OpConstructorMap variables;
  std::unordered_map<std::string, MlirWrapperBase *> decorators;
};
