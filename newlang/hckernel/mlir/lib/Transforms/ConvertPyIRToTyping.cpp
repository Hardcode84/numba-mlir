// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <unordered_map>
#include <unordered_set>

#include "hc/Utils.hpp"

namespace hc {
#define GEN_PASS_DEF_CONVERTLOADVARTYPINGPASS
#define GEN_PASS_DEF_CONVERTFUNCTYPINGPASS
#define GEN_PASS_DEF_INLINEFORCEINLINEDFUNCSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ConvertLoadVarToMakeIdent final
    : public mlir::OpRewritePattern<hc::py_ir::LoadVarOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  ConvertLoadVarToMakeIdent(mlir::MLIRContext *context,
                            const hc::OpConstructorMap &opConstrMap)
      : OpRewritePattern(context), opConstrMap(opConstrMap) {}

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::LoadVarOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto name = op.getName().str();

    auto *context = getContext();
    auto strAttr = [&rewriter](llvm::StringRef str) {
      return rewriter.getStringAttr(str);
    };
    auto arrAttr = [&rewriter](llvm::ArrayRef<mlir::Attribute> arr) {
      return rewriter.getArrayAttr(arr);
    };

    auto opConstr = opConstrMap.find(name);
    if (opConstr != opConstrMap.end()) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      rewriter.setInsertionPoint(op);
      auto *newOp = opConstr->second(rewriter);
      rewriter.replaceOp(op, newOp);

      return mlir::success();
    }

    return mlir::failure();
  }

  const hc::OpConstructorMap &opConstrMap;
};

struct ReplaceCallWithOp final
    : public mlir::OpRewritePattern<hc::py_ir::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcType = op.getFunc().getType();
    auto *context = rewriter.getContext();
    if (auto ident = mlir::dyn_cast<hc::typing::IdentType>(funcType)) {
      auto name = ident.getName();

      if (name == "typing.check") {
        rewriter.setInsertionPoint(op);
        rewriter.create<hc::typing::CheckOp>(op.getLoc(), op.getArgs()[0]);
        rewriter.eraseOp(op);

        return mlir::success();
      }

      if (name == "typing.is_same") {
        auto valOrCast = [&](mlir::Value val) {
          auto type = val.getType();

          if (mlir::isa<hc::typing::ValueType>(type))
            return val;

          rewriter.setInsertionPoint(op);
          return rewriter
              .create<hc::typing::CastOp>(
                  op.getLoc(), hc::typing::ValueType::get(context), val)
              .getResult();
        };
        rewriter.replaceOpWithNewOp<hc::typing::IsSameOp>(
            op, rewriter.getI1Type(), valOrCast(op.getArgs()[0]),
            valOrCast(op.getArgs()[1]));

        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

static void replaceReturns(mlir::PatternRewriter &rewriter,
                           hc::py_ir::PyFuncOp func) {
  llvm::SmallVector<hc::py_ir::ReturnOp> returns;
  func->walk([&](hc::py_ir::ReturnOp ret) { returns.push_back(ret); });

  for (auto &&ret : returns) {
    rewriter.setInsertionPoint(ret);
    rewriter.replaceOpWithNewOp<hc::typing::TypeResolverReturnOp>(
        ret, ret.getOperand());
  }
}

static void updateModEnd(mlir::PatternRewriter &rewriter,
                         hc::py_ir::PyFuncOp func) {
  auto users = func->getUsers();
  auto modEnd = mlir::dyn_cast<hc::py_ir::PyModuleEndOp>(*users.begin());
  assert(modEnd);

  // Is this safe?
  rewriter.modifyOpInPlace(modEnd, [&]() {
    auto vals = modEnd.getResultsMutable();

    for (auto &&operand : vals) {
      if (operand.get() == func.getResult()) {
        vals.erase(operand.getOperandNumber());
        break;
      }
    }
  });
}

static void replaceArgsWithGetArg(mlir::PatternRewriter &rewriter,
                                  hc::py_ir::PyFuncOp func) {
  auto *entryBlock = func.getEntryBlock();
  auto args = func.getBlockArgs();
  auto *context = rewriter.getContext();

  rewriter.setInsertionPointToStart(entryBlock);
  auto numArgs = args.size();
  for (int i = 0; i < numArgs; ++i) {
    auto arg = args[i];
    auto loc = arg.getLoc();
    auto ind = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
    auto getArg = rewriter.create<hc::typing::GetArgOp>(
        loc, hc::typing::ValueType::get(context), ind);
    rewriter.replaceAllUsesWith(arg, getArg);
  }
}

static void inlineCapturedArgs(mlir::PatternRewriter &rewriter,
                               hc::py_ir::PyFuncOp func) {
  auto capturesOperands = func.getCaptureArgs();
  auto capturesArgs = func.getCaptureBlockArgs();
  auto capturesNum = capturesArgs.size();
  for (int i = 0; i < capturesNum; ++i) {
    auto capt = capturesArgs[i];
    if (!capt.use_empty()) {
      auto *defOp = capturesOperands[i].getDefiningOp();
      auto *clone = rewriter.insert(defOp->clone());
      // what to do if there is more than one result?
      auto newVal = clone->getResult(0);
      rewriter.replaceAllUsesWith(capt, newVal);
    }
  }
}

static void processArgs(mlir::PatternRewriter &rewriter,
                        hc::py_ir::PyFuncOp func) {
  replaceArgsWithGetArg(rewriter, func);
  inlineCapturedArgs(rewriter, func);

  auto *entryBlock = func.getEntryBlock();
  entryBlock->eraseArguments([](auto barg) { return barg.use_empty(); });
}

static void reaplaceWithResolver(mlir::PatternRewriter &rewriter,
                                 hc::py_ir::PyFuncOp func) {
  auto strAttr = [&rewriter](llvm::StringRef str) {
    return rewriter.getStringAttr(str);
  };
  auto arrAttr = [&rewriter](llvm::ArrayRef<mlir::Attribute> arr) {
    return rewriter.getArrayAttr(arr);
  };

  auto name = func.getName();
  auto loc = func.getLoc();

  auto resAttr = arrAttr({strAttr(name)});
  rewriter.setInsertionPoint(func);
  auto resolver = rewriter.create<hc::typing::TypeResolverOp>(loc, resAttr);
  auto &region = resolver.getBodyRegion();
  region.takeBody(func.getBodyRegion());

  rewriter.eraseOp(func);
}

struct ReplaceFuncWithResolver final
    : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto users = op->getUsers();

    if (op->hasOneUse() &&
        mlir::isa<hc::py_ir::PyModuleEndOp>(*users.begin())) {
      processArgs(rewriter, op);
      replaceReturns(rewriter, op);
      updateModEnd(rewriter, op);
      reaplaceWithResolver(rewriter, op);

      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ConvertFuncTypingPass final
    : public hc::impl::ConvertFuncTypingPassBase<ConvertFuncTypingPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    hc::populateConvertFuncTypingPatterns(patterns);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

struct ConvertLoadvarTypingPass
    : public mlir::PassWrapper<ConvertLoadvarTypingPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLoadvarTypingPass)

  ConvertLoadvarTypingPass(const hc::OpConstructorMap &opConstrMap)
      : opConstrMap(opConstrMap) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::hc::py_ir::PyIRDialect>();
    registry.insert<::mlir::cf::ControlFlowDialect>();
    registry.insert<::mlir::arith::ArithDialect>();
    registry.insert<::hc::typing::TypingDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    hc::populateConvertLoadvarTypingPatterns(patterns, opConstrMap);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

  const hc::OpConstructorMap &opConstrMap;
};

struct CleanupUnusedCaptures final
    : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto capturesBlockArgs = op.getCaptureBlockArgs();
    auto captureNames = op.getCaptureNames();

    bool hasUnused = false;
    for (auto &&arg : capturesBlockArgs) {
      if (arg.use_empty()) {
        hasUnused = true;
        break;
      }
    }

    if (!hasUnused)
      return mlir::failure();

    mlir::SmallVector<unsigned> indexes;
    for (int i = captureNames.size() - 1; i >= 0; --i) {
      auto capt = capturesBlockArgs[i];
      if (capt.use_empty()) {
        indexes.push_back(i);
      }
    }

    auto eraseCaptureByIndex = [&](unsigned idx) {
      int blockArgIdx = idx + op.getBlockArgs().size();
      auto *entryBlock = op.getEntryBlock();
      op.getCaptureArgsMutable().erase(idx);
      auto captNames = mlir::SmallVector<mlir::Attribute>(
          op.getCaptureNames().getAsRange<mlir::Attribute>());
      captNames.erase(captNames.begin() + idx);
      auto attr =
          rewriter.getArrayAttr(mlir::ArrayRef<mlir::Attribute>(captNames));
      op.setCaptureNamesAttr(attr);
      entryBlock->eraseArgument(blockArgIdx);
    };

    for (auto &&idx : indexes) {
      eraseCaptureByIndex(idx);
    }

    return mlir::success();
  }
};

struct PropagateCapturesFuncs final
    : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto captures = op.getCaptureArgs();
    auto capturesBlockArgs = op.getCaptureBlockArgs();

    bool updated = false;
    for (int i = 0; i < captures.size(); ++i) {
      auto capt = captures[i];
      auto blockArg = capturesBlockArgs[i];
      auto defOp = capt.getDefiningOp();
      if (auto capturedFunc = mlir::dyn_cast<hc::py_ir::PyFuncOp>(defOp)) {
        if (capturedFunc->hasAttr("force_inline")) {
          if (!blockArg.use_empty()) {
            rewriter.replaceAllUsesWith(blockArg, capt);
            updated = true;
          }
        }
      }
    }

    return mlir::success(updated);
  }
};

struct InlineForceInlinedPass final
    : public hc::impl::InlineForceInlinedFuncsPassBase<InlineForceInlinedPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    hc::populateInlineForceInlinedPatterns(patterns);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
hc::createConvertLoadvarTypingPass(const OpConstructorMap &opConstrMap) {
  return std::make_unique<ConvertLoadvarTypingPass>(opConstrMap);
}

void hc::populateConvertLoadvarTypingPatterns(
    mlir::RewritePatternSet &patterns, const OpConstructorMap &opConstrMap) {
  patterns.insert<ConvertLoadVarToMakeIdent>(patterns.getContext(),
                                             opConstrMap);
}

void hc::populateConvertFuncTypingPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ReplaceFuncWithResolver, ReplaceCallWithOp>(
      patterns.getContext());
}

void hc::populateInlineForceInlinedPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<PropagateCapturesFuncs, CleanupUnusedCaptures>(
      patterns.getContext());
}
