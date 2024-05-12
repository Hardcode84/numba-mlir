// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include "hc/PyFront/Import.hpp"
#include "hc/Transforms/Passes.hpp"
#include "hc/Utils.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static void printDiag(llvm::raw_ostream &os, const mlir::Diagnostic &diag) {
  os << diag;
  for (auto &note : diag.getNotes())
    os << "\n" << note;

  os << "\n";
}

static mlir::LogicalResult runUnderDiag(mlir::PassManager &pm,
                                        mlir::ModuleOp module) {
  bool dumpDiag = true;
  std::string err;
  llvm::raw_string_ostream errStream(err);
  auto diagHandler = [&](const mlir::Diagnostic &diag) {
    if (dumpDiag)
      printDiag(llvm::errs(), diag);

    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
      printDiag(errStream, diag);
  };

  auto getErr = [&]() -> const std::string & {
    errStream << "\n";
    module.print(errStream);
    errStream.flush();
    return err;
  };

  bool verify = true;
  return hc::scopedDiagHandler(*module.getContext(), diagHandler, [&]() {
    if (verify && mlir::failed(mlir::verify(module))) {
      llvm::errs() << "MLIR broken module\n" << getErr();
      return mlir::failure();
    }

    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "MLIR pipeline failed\n" << getErr();
      return mlir::failure();
    }

    return mlir::success();
  });
}

static void populatePyIROptPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "PyIROptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(mlir::createCSEPass());
        p.addPass(hc::createCleanupPySetVarPass());
      }));
}

static void populatePasses(mlir::PassManager &pm) {
  pm.addPass(hc::createSimplifyASTPass());
  pm.addPass(hc::createConvertPyASTToIRPass());
  populatePyIROptPasses(pm);
  pm.addPass(hc::createReconstuctPySSAPass());
  populatePyIROptPasses(pm);
  pm.addPass(hc::createPyTypeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static bool compile_ast(std::string source) {
  mlir::MLIRContext ctx;
  auto loc = mlir::OpBuilder(&ctx).getUnknownLoc();

  auto mod = mlir::ModuleOp::create(loc);

  auto res = hc::importPyModule(source, mod);

  if (mlir::succeeded(res)) {
    mlir::PassManager pm(&ctx);

    ctx.disableMultithreading();
    pm.enableIRPrinting();

    populatePasses(pm);
    if (mlir::failed(runUnderDiag(pm, mod)))
      return false;

    return true;
  }

  return true;
}

PYBIND11_MODULE(compiler, m) {
  m.def("compile_ast", &compile_ast, "compile_ast", py::arg("source"));
}
