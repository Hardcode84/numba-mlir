// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingDispatcher.hpp"

#include <mlir/CAPI/IR.h>
#include <mlir/IR/Operation.h>

#include "Utils.hpp"

#include "hc/Pipelines/FrontendPipeline.hpp"
#include "hc/Transforms/ModuleLinker.hpp"

#include "IRModule.h"

namespace py = pybind11;

static void linkModules(MlirModule dest, MlirModule toLink) {
  mlir::ModuleOp destMod = unwrap(dest);
  mlir::ModuleOp toLinkMod = unwrap(toLink);
  if (mlir::failed(hc::linkModules(destMod, toLinkMod)))
    reportError("Failed to link modules");
}

void TypingDispatcher::definePyClass(py::module_ &m) {
  py::class_<TypingDispatcher>(m, "TypingDispatcher")
      .def(py::init<py::capsule, py::object>())
      .def("compile", &TypingDispatcher::compile);

  m.def("link_modules", &linkModules);
}

py::object TypingDispatcher::compile() {
  mlir::OwningOpRef op = mlir::cast<mlir::ModuleOp>(importFunc()->clone());
  auto res = mlir::python::PyModule::forModule(wrap(op.get()));
  op.release();
  return res.getObject();
}

void TypingDispatcher::populateImportPipeline(mlir::PassManager &pm) {
  hc::populateFrontendPipeline(pm);
}

void TypingDispatcher::populateInvokePipeline(mlir::PassManager &pm) {
  reportError("TODO: compile");
}
