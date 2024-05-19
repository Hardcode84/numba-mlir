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

    populateArgsHandlers(desc.attr("args"));
  }

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

using HandlerT = std::function<void(mlir::MLIRContext &, pybind11::handle,
                                    llvm::SmallVectorImpl<mlir::Type> &,
                                    llvm::SmallVectorImpl<PyObject *> &)>;

static HandlerT getArgHandler(py::handle arg) {
  py::str sym("sym");
  if (arg.equal(sym)) {
    return [](mlir::MLIRContext &ctx, py::handle obj,
              llvm::SmallVectorImpl<mlir::Type> &ret,
              llvm::SmallVectorImpl<PyObject *> &args) {
      if (py::isinstance<py::int_>(obj)) {
        ret.emplace_back(mlir::IntegerType::get(&ctx, 64));
      } else if (py::isinstance<py::float_>(obj)) {
        ret.emplace_back(mlir::Float64Type::get(&ctx));
      } else {
        reportError(llvm::Twine("Unsupported type") +
                    py::str(obj).cast<std::string>());
      }
      args.emplace_back(obj.ptr());
    };
  }
  if (py::isinstance<py::tuple>(arg)) {
    auto count = py::len(arg);
    llvm::SmallVector<HandlerT, 0> handlers;
    handlers.reserve(count);
    for (auto elem : arg)
      handlers.emplace_back(getArgHandler(elem));

    return [handlersCopy =
                std::move(handlers)](mlir::MLIRContext &ctx, py::handle obj,
                                     llvm::SmallVectorImpl<mlir::Type> &ret,
                                     llvm::SmallVectorImpl<PyObject *> &args) {
      for (auto &&[h, elem] : llvm::zip_equal(handlersCopy, obj))
        h(ctx, elem, ret, args);
    };
  }
  if (py::isinstance<py::list>(arg)) {
    reportError(llvm::Twine("TODO: handle buffer"));
  }

  reportError(llvm::Twine("Unsupported arg type") +
              py::str(arg).cast<std::string>());
}

void Dispatcher::populateArgsHandlers(pybind11::handle args) {
  auto &ctx = context.context;
  assert(argsHandlers.empty());
  argsHandlers.reserve(py::len(args));
  for (auto [name, elem] : args.cast<py::dict>()) {
    auto nameAttr = mlir::StringAttr::get(&ctx, name.cast<std::string>());
    auto handler = getArgHandler(elem);
    argsHandlers.emplace_back(ArgDesc{nameAttr.getValue(), std::move(handler)});
  }
}

mlir::Type
Dispatcher::processArgs(py::args &args, py::kwargs &kwargs,
                        llvm::SmallVectorImpl<PyObject *> &retArgs) const {
  auto srcNumArgs = args.size();
  bool hasKWArgs = kwargs.size() > 0;
  auto getKWArg = [&](llvm::StringRef name) -> py::handle {
    if (!hasKWArgs)
      return nullptr;

    py::str n(name.data(), name.size());
    if (kwargs.contains(n))
      return kwargs[n];

    return nullptr;
  };

  llvm::SmallVector<mlir::Type> types;
  types.reserve(argsHandlers.size());

  auto &ctx = context.context;
  size_t idx = 0;
  for (auto &arg : argsHandlers) {
    auto name = arg.name;
    if (auto kwarg = getKWArg(name)) {
      arg.handler(ctx, kwarg, types, retArgs);
      continue;
    }
    if (idx >= srcNumArgs)
      reportError("Insufficient args");

    auto srcArg = args[idx++];
    arg.handler(context.context, srcArg, types, retArgs);
  }

  if (types.empty())
    return nullptr;

  if (types.size() == 1)
    return types.front();

  return mlir::TupleType::get(&ctx);
}
