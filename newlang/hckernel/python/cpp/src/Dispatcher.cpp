// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/Support/LogicalResult.h>

#include "CompilerFront.hpp"
#include "Context.hpp"

namespace py = pybind11;

[[noreturn]] static void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}

void Dispatcher::definePyClass(py::module_ &m) {
  py::class_<Dispatcher>(m, "Dispatcher")
      .def(py::init<py::capsule, py::object>())
      .def("__call__", &Dispatcher::call);
}

Dispatcher::Dispatcher(py::capsule ctx, py::object getSrc)
    : context(*ctx.get_pointer<Context>()), contextRef(std::move(ctx)),
      getSourceFunc(std::move(getSrc)) {}

static std::pair<std::string, std::string> getSource(py::object getSourceFunc) {
  auto res = getSourceFunc().cast<py::tuple>();
  return {res[0].cast<std::string>(), res[1].cast<std::string>()};
}

void Dispatcher::call(py::args args, py::kwargs kwargs) {
  if (!func) {
    assert(getSourceFunc);
    auto [src, funcName] = getSource(std::move(getSourceFunc));
    if (mlir::failed(compileAST(context.context, src, funcName)))
      reportError("Compilation failed");
  }

  assert(func && "Func is not set");

  ExceptionDesc exc;
  if (func(&exc, args.ptr(), kwargs.ptr()) != 0)
    reportError(exc.message);
}
