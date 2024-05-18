// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/IR/BuiltinOps.h>
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

Dispatcher::Dispatcher(py::capsule ctx, py::object getDesc)
    : context(*ctx.get_pointer<Context>()), contextRef(std::move(ctx)),
      getFuncDesc(std::move(getDesc)) {}

Dispatcher::~Dispatcher() {}

static std::pair<std::string, std::string> getSource(py::handle desc) {
  return {desc.attr("source").cast<std::string>(),
          desc.attr("name").cast<std::string>()};
}

void Dispatcher::call(py::args args, py::kwargs kwargs) {
  if (!mod) {
    assert(getFuncDesc);
    py::object desc = getFuncDesc();
    getFuncDesc = py::object();
    auto [src, funcName] = getSource(desc);
    auto res = compileAST(context.context, src, funcName);
    if (mlir::failed(res))
      reportError("Compilation failed");
    std::swap(mod, *res);
  }

  assert(func && "Func is not set");

  ExceptionDesc exc;
  if (func(&exc, args.ptr(), kwargs.ptr()) != 0)
    reportError(exc.message);
}
