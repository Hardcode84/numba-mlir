// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include <llvm/ADT/Twine.h>

#include "Utils.hpp"

#include "hc/Pipelines/FrontendPipeline.hpp"

namespace py = pybind11;

void Dispatcher::definePyClass(py::module_ &m) {
  py::class_<Dispatcher>(m, "Dispatcher")
      .def(py::init<py::capsule, py::object>())
      .def("__call__", &Dispatcher::call);
}

void Dispatcher::call(py::args args, py::kwargs kwargs) {
  importFunc();
  invokeFunc(args, kwargs);
}

void Dispatcher::populateImportPipeline(mlir::PassManager &pm) {
  hc::populateFrontendPipeline(pm);
}

void Dispatcher::populateInvokePipeline(mlir::PassManager &pm) {
  reportError("TODO: compile");
}
