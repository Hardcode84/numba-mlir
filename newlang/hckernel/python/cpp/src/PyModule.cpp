// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Context.hpp"
#include "Dispatcher.hpp"
#include "MlirWrappers.hpp"
#include "TypingDispatcher.hpp"

namespace py = pybind11;

PYBIND11_MODULE(compiler, m) {
  m.def("create_context", &createContext);

  Dispatcher::definePyClass(m);
  TypingDispatcher::definePyClass(m);
  MlirWrapperBase::definePyClass(m);
  MlirOpWrapper::definePyClass(m);
  MlirTypeWrapper::definePyClass(m);
  MlirDecoratorWrapper::definePyClass(m);

  addBuildinTypes(m);
  addBuildinOps(m);
}
