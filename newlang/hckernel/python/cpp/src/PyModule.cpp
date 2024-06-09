// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Context.hpp"
#include "Dispatcher.hpp"
#include "PyWrappers.hpp"

namespace py = pybind11;

PYBIND11_MODULE(compiler, m) {
  m.def("create_context", &createContext);

  Dispatcher::definePyClass(m);

  auto mlir_mod = m.def_submodule("_mlir");
  populateMlirModule(mlir_mod);
}
