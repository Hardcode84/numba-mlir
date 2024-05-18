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

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"
#include "hc/Pipelines/FrontendPipeline.hpp"
#include "hc/PyFront/Import.hpp"
#include "hc/Utils.hpp"

#include "CompilerFront.hpp"

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

static mlir::LogicalResult importAST(mlir::ModuleOp mod, llvm::StringRef source,
                                     llvm::StringRef funcName) {
  auto res = hc::importPyModule(source, mod);
  if (mlir::failed(res))
    return mlir::failure();

  auto pyMod = mlir::cast<hc::py_ast::PyModuleOp>(*res);

  mlir::OpBuilder builder(mod.getContext());
  auto term = pyMod.getBody()->getTerminator();
  builder.setInsertionPoint(term);
  builder.create<hc::py_ast::CaptureValOp>(term->getLoc(), funcName);
  return mlir::success();
}

mlir::LogicalResult compileAST(mlir::MLIRContext &ctx, llvm::StringRef source,
                               llvm::StringRef funcName) {
  auto loc = mlir::OpBuilder(&ctx).getUnknownLoc();

  auto mod = mlir::ModuleOp::create(loc);

  auto res = importAST(mod, source, funcName);
  if (mlir::failed(res))
    return mlir::failure();

  mlir::PassManager pm(&ctx);

  ctx.disableMultithreading();
  pm.enableIRPrinting();

  hc::populateFrontendPipeline(pm);
  return runUnderDiag(pm, mod);
}
