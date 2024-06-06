// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include "hc/Pipelines/FrontendPipeline.hpp"

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
    llvm::SmallVector<ImportedSym> symbols;
    for (auto it : desc.attr("imported_symbols")) {
      auto elem = it.cast<py::tuple>();
      ImportedSym sym;
      sym.name = elem[0].cast<std::string>();
      for (auto path : elem[1])
        sym.modulePath.emplace_back(path.cast<std::string>());

      symbols.emplace_back(std::move(sym));
    }
    auto res = compileAST(context, src, funcName, symbols);
    if (mlir::failed(res))
      reportError("AST import failed");

    auto *mlirContext = &context.context;
    mlir::PassManager pm(mlirContext);

    if (context.settings.dumpIR) {
      mlirContext->disableMultithreading();
      pm.enableIRPrinting();
    }

    hc::populateFrontendPipeline(pm);
    if (mlir::failed(runUnderDiag(pm, **res)))
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

static const constexpr int64_t kDynamic = mlir::ShapedType::kDynamic;
static const constexpr int64_t kLiteral = kDynamic + 1;

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
    py::str lit("lit");
    llvm::SmallVector<int64_t> srcShape(py::len(arg));
    for (auto &&[i, s] : llvm::enumerate(arg)) {
      if (py::isinstance<py::int_>(s)) {
        srcShape[i] = s.cast<int64_t>();
      } else if (s.equal(sym)) {
        srcShape[i] = kDynamic;
      } else if (s.equal(lit)) {
        srcShape[i] = kLiteral;
      } else {
        reportError(llvm::Twine("Unsupported dim type: ") +
                    py::str(s).cast<std::string>());
      }
    }

    return [argShape =
                std::move(srcShape)](mlir::MLIRContext &ctx, py::handle obj,
                                     llvm::SmallVectorImpl<mlir::Type> &ret,
                                     llvm::SmallVectorImpl<PyObject *> &args) {
      auto arrInterface = obj.attr("__array_interface__").cast<py::dict>();
      auto shape = arrInterface["shape"].cast<py::tuple>();
      if (argShape.size() != shape.size())
        reportError("Invalid buffer rank");

      llvm::SmallVector<int64_t> resShape(argShape.size());
      for (auto &&[i, s] : llvm::enumerate(argShape)) {
        if (s == kDynamic) {
          resShape[i] = kDynamic;
        } else if (s == kLiteral) {
          resShape[i] = shape[i].cast<int64_t>();
        } else {
          resShape[i] = s;
        }
      }

      // TODO: dtype
      // TODO: layout
      auto dtype = mlir::Float64Type::get(&ctx);
      ret.emplace_back(mlir::MemRefType::get(resShape, dtype));
      args.emplace_back(obj.ptr());
    };
  }

  reportError(llvm::Twine("Unsupported arg type: ") +
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
