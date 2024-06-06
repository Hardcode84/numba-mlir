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
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Pipelines/FrontendPipeline.hpp"
#include "hc/PyFront/Import.hpp"
#include "hc/Utils.hpp"

#include "CompilerFront.hpp"
#include "Context.hpp"

static void printDiag(llvm::raw_ostream &os, const mlir::Diagnostic &diag) {
  os << diag;
  for (auto &note : diag.getNotes())
    os << "\n" << note;

  os << "\n";
}

mlir::LogicalResult runUnderDiag(mlir::PassManager &pm,
                                 mlir::Operation *module) {
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
    module->print(errStream);
    errStream.flush();
    return err;
  };

  bool verify = true;
  return hc::scopedDiagHandler(*module->getContext(), diagHandler, [&]() {
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

static void createModuleImport(mlir::OpBuilder &builder,
                               const ImportedSym &sym) {
  assert(!sym.modulePath.empty());
  mlir::Value mod;

  mlir::Location loc = builder.getUnknownLoc();
  auto type = hc::py_ir::UndefinedType::get(builder.getContext());
  for (auto &&path : sym.modulePath) {
    if (!mod) {
      mod = builder.create<hc::py_ir::LoadModuleOp>(loc, type, path);
      continue;
    }

    mod = builder.create<hc::py_ir::GetAttrOp>(loc, type, mod, path);
  }
  assert(mod);
  builder.create<hc::py_ir::StoreVarOp>(loc, sym.name, mod);
}

static mlir::LogicalResult
importAST(mlir::Operation *mod, llvm::StringRef source,
          llvm::StringRef funcName, llvm::ArrayRef<ImportedSym> importedSymbols,
          bool dumpAST) {
  auto res = hc::importPyModule(source, mod, dumpAST);
  if (mlir::failed(res))
    return mlir::failure();

  auto pyMod = mlir::cast<hc::py_ast::PyModuleOp>(*res);

  mlir::OpBuilder builder(mod->getContext());
  mlir::Block *body = pyMod.getBody();

  builder.setInsertionPointToStart(body);
  for (auto &&sym : importedSymbols)
    createModuleImport(builder, sym);

  auto term = body->getTerminator();
  builder.setInsertionPoint(term);
  builder.create<hc::py_ast::CaptureValOp>(term->getLoc(), funcName);
  return mlir::success();
}

mlir::FailureOr<mlir::OwningOpRef<mlir::Operation *>>
compileAST(Context &ctx, llvm::StringRef source, llvm::StringRef funcName,
           llvm::ArrayRef<ImportedSym> importedSymbols) {
  auto *mlirContext = &ctx.context;
  mlirContext->loadDialect<hc::py_ir::PyIRDialect>();
  auto loc = mlir::OpBuilder(mlirContext).getUnknownLoc();

  mlir::OwningOpRef<mlir::Operation *> mod(mlir::ModuleOp::create(loc));

  auto &settings = ctx.settings;
  if (mlir::failed(
          importAST(*mod, source, funcName, importedSymbols, settings.dumpAST)))
    return mlir::failure();

  mlir::PassManager pm(mlirContext);

  if (settings.dumpIR) {
    mlirContext->disableMultithreading();
    pm.enableIRPrinting();
  }

  hc::populateFrontendPipeline(pm);
  if (mlir::failed(runUnderDiag(pm, *mod)))
    return mlir::failure();

  return mod;
}
