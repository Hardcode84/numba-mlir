// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include <llvm/ADT/Twine.h>

namespace py = pybind11;

[[noreturn]] static void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}

void Dispatcher::definePyClass(py::module_ &m) {
  py::class_<Dispatcher>(m, "Dispatcher")
      .def(py::init<py::capsule, py::object>())
      .def("__call__", &Dispatcher::call);
}

void Dispatcher::call(py::args args, py::kwargs kwargs) {
  importFunc();

  llvm::SmallVector<PyObject *, 16> funcArgs;
  mlir::Type key = processArgs(args, kwargs, funcArgs);
  auto it = funcsCache.find(key);
  if (it == funcsCache.end()) {
    reportError("TODO: compile");
    it = funcsCache.insert({key, nullptr}).first;
  }

  auto func = it->second;
  ExceptionDesc exc;
  if (func(&exc, funcArgs.data()) != 0)
    reportError(exc.message);
}
