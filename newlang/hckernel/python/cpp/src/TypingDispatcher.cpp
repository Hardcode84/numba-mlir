// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingDispatcher.hpp"

#include "Utils.hpp"

#include "hc/Pipelines/FrontendPipeline.hpp"

namespace py = pybind11;

void TypingDispatcher::definePyClass(py::module_ &m) {
  py::class_<TypingDispatcher>(m, "TypingDispatcher")
      .def(py::init<py::capsule, py::object>())
      .def("compile", &TypingDispatcher::compile);
}

void TypingDispatcher::compile() { importFunc(); }

void TypingDispatcher::populateImportPipeline(mlir::PassManager &pm) {
  hc::populateFrontendPipeline(pm);
}

void TypingDispatcher::populateInvokePipeline(mlir::PassManager &pm) {
  reportError("TODO: compile");
}
