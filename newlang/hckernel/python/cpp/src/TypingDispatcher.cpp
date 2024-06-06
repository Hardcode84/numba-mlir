// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingDispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include "CompilerFront.hpp"
#include "Context.hpp"
#include "MlirWrappers.hpp"
#include "hc/Pipelines/FrontendPipeline.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include "DecoratorPipeline.hpp"

#include <unordered_map>
#include <vector>

namespace py = pybind11;

[[noreturn]] static void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}

void TypingDispatcher::definePyClass(py::module_ &m) {
  py::class_<TypingDispatcher>(m, "TypingDispatcher")
      .def(py::init<py::capsule, py::object, py::dict>())
      .def("compile", &TypingDispatcher::compile);
}

TypingDispatcher::TypingDispatcher(py::capsule ctx, py::object getDesc,
                                   pybind11::dict globals)
    : context(*ctx.get_pointer<Context>()), contextRef(std::move(ctx)),
      getFuncDesc(std::move(getDesc)), globals(globals) {}

TypingDispatcher::~TypingDispatcher() {}

static std::pair<std::string, std::string> getSource(py::handle desc) {
  return {desc.attr("source").cast<std::string>(),
          desc.attr("name").cast<std::string>()};
}

void TypingDispatcher::compile() {
  if (!mod) {
    assert(getFuncDesc);
    py::object desc = getFuncDesc();
    getFuncDesc = py::object();
    auto [src, funcName] = getSource(desc);

    llvm::SmallVector<ImportedSym> symbols;
    auto res = compileAST(context, src, funcName, symbols);
    if (mlir::failed(res))
      reportError("AST import failed");

    auto *mlirContext = &context.context;
    mlir::PassManager pm(mlirContext);

    if (context.settings.dumpIR) {
      mlirContext->disableMultithreading();
      pm.enableIRPrinting();
    }

    hc::populatePyIRPipeline(pm);
    if (mlir::failed(runUnderDiag(pm, **res)))
      reportError("Compilation failed");

    std::swap(mod, *res);

    DecoratorPipeline pipeline(context);
    mod->walk<mlir::WalkOrder::PreOrder>([&](hc::py_ir::LoadVarOp loadVar) {
      auto name = loadVar.getName().str();
      auto var = globals[name.c_str()];
      if (!py::isinstance<MlirWrapperBase>(var)) {
        reportError("Variable is absent or of wrong type");
      }
      auto *mwVar = var.cast<MlirWrapperBase *>();
      mwVar->updatePipeline(name, &pipeline);
    });

    pipeline.run(*mod);
  }
}
