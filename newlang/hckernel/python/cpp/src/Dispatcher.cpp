// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include "CompilerFront.hpp"

namespace py = pybind11;

Dispatcher::Dispatcher(py::object getSrc) : getSourceFunc(std::move(getSrc)) {}

static std::pair<std::string, std::string> getSource(py::handle getSourceFunc) {
  auto res = getSourceFunc().cast<py::tuple>();
  return {res[0].cast<std::string>(), res[1].cast<std::string>()};
}

void Dispatcher::call(py::args args, py::kwargs kwargs) {
  auto [src, funcName] = getSource(getSourceFunc);
  getSourceFunc = py::object();
  compileAST(src, funcName);
}
