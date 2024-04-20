// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/Dialect/Complex/IR/Complex.h>
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

enum class Cmd {
  Front,
  Full,
};

using ChunkBufferHandler = llvm::function_ref<mlir::LogicalResult(
    std::unique_ptr<llvm::MemoryBuffer> chunkBuffer, llvm::raw_ostream &os)>;

static mlir::LogicalResult
splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> originalBuffer,
                      ChunkBufferHandler processChunkBuffer,
                      llvm::raw_ostream &os, bool enableSplitting = true,
                      bool insertMarkerInOutput = false,
                      llvm::StringRef splitMarker = "// -----") {
  using namespace mlir;
  // If splitting is disabled, we process the full input buffer.
  if (!enableSplitting)
    return processChunkBuffer(std::move(originalBuffer), os);

  const int splitMarkerLen = splitMarker.size();

  auto *origMemBuffer = originalBuffer.get();
  SmallVector<StringRef, 8> rawSourceBuffers;
  const int checkLen = 2;
  // Split dropping the last checkLen chars to enable flagging near misses.
  origMemBuffer->getBuffer().split(rawSourceBuffers,
                                   splitMarker.drop_back(checkLen));
  if (rawSourceBuffers.empty())
    return success();

  // Add the original buffer to the source manager.
  llvm::SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(originalBuffer), SMLoc());

  // Flag near misses by iterating over all the sub-buffers found when splitting
  // with the prefix of the splitMarker. Use a sliding window where we only add
  // a buffer as a sourceBuffer if terminated by a full match of the
  // splitMarker, else flag a warning (if near miss) and extend the size of the
  // buffer under consideration.
  SmallVector<StringRef, 8> sourceBuffers;
  StringRef prev;
  for (auto buffer : rawSourceBuffers) {
    if (prev.empty()) {
      prev = buffer;
      continue;
    }

    // Check that suffix is as expected and doesn't have any dash post.
    bool expectedSuffix = buffer.starts_with(splitMarker.take_back(checkLen)) &&
                          buffer.size() > checkLen && buffer[checkLen] != '0';
    if (expectedSuffix) {
      sourceBuffers.push_back(prev);
      prev = buffer.drop_front(checkLen);
    } else {
      // TODO: Consider making this a failure.
      auto splitLoc = SMLoc::getFromPointer(buffer.data());
      fileSourceMgr.PrintMessage(llvm::errs(), splitLoc,
                                 llvm::SourceMgr::DK_Warning,
                                 "near miss with file split marker");
      prev = StringRef(prev.data(),
                       prev.size() + splitMarkerLen - checkLen + buffer.size());
    }
  }
  if (!prev.empty())
    sourceBuffers.push_back(prev);

  // Process each chunk in turn.
  bool hadFailure = false;
  auto interleaveFn = [&](StringRef subBuffer) {
    auto splitLoc = SMLoc::getFromPointer(subBuffer.data());
    unsigned splitLine = fileSourceMgr.getLineAndColumn(splitLoc).first;
    auto subMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(
        subBuffer, Twine("within split at ") +
                       origMemBuffer->getBufferIdentifier() + ":" +
                       Twine(splitLine) + " offset ");
    if (failed(processChunkBuffer(std::move(subMemBuffer), os)))
      hadFailure = true;
  };

  llvm::SmallString<64> sep;
  if (insertMarkerInOutput) {
    sep.append("\n");
    sep.append(splitMarker);
    sep.append("\n");
  }

  llvm::interleave(sourceBuffers, os, interleaveFn, sep);

  // If any fails, then return a failure of the tool.
  return failure(hadFailure);
}

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

static mlir::LogicalResult pyfrontMain(llvm::StringRef inputFilename, Cmd cmd) {
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::complex::ComplexDialect>();
  auto loc = mlir::OpBuilder(&ctx).getUnknownLoc();

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                           llvm::raw_ostream &os) -> mlir::LogicalResult {
    auto mod = mlir::ModuleOp::create(loc);
    auto res = hc::importPyModule(chunkBuffer->getBuffer(), mod);
    if (mlir::succeeded(res)) {
      if (mlir::failed(mlir::verify(mod)))
        res = mlir::failure();

      mod.print(os);

      if (cmd == Cmd::Full) {
        mlir::PassManager pm(&ctx);

        ctx.disableMultithreading();
        pm.enableIRPrinting();

        populatePasses(pm);
        if (mlir::failed(runUnderDiag(pm, mod)))
          return mlir::failure();
      }
    }

    mod->erase();
    return res;
  };

  return splitAndProcessBuffer(std::move(file), processBuffer, llvm::outs(),
                               true, false, "# -----");
}

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << argv[0] << " front|full <filename>";
    return EXIT_FAILURE;
  }

  llvm::StringRef cmd = argv[1];
  Cmd cmdId;
  if (cmd == "front") {
    cmdId = Cmd::Front;
  } else if (cmd == "full") {
    cmdId = Cmd::Full;
  } else {
    llvm::errs() << "Unknown command: " << cmd;
    return EXIT_FAILURE;
  }

  llvm::StringRef file = argv[2];

  return mlir::asMainReturnCode(pyfrontMain(file, cmdId));
}
