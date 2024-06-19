// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingPipeline.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Transforms/Passes.hpp"

static void convertCall(mlir::IRRewriter &builder, hc::py_ir::CallOp call,
                        llvm::StringRef name) {
  builder.setInsertionPoint(call);
  mlir::Type callResType = call.getResult().getType();
  mlir::ValueRange args = call.getArgs();
  mlir::TypeRange callArgTypes = args.getTypes();
  auto checkCall = [&](llvm::StringRef funcName, mlir::Type resType,
                       mlir::TypeRange argTypes) -> bool {
    if (funcName != name)
      return false;

    if (callResType != resType)
      return false;

    return llvm::equal(callArgTypes, argTypes);
  };

  auto index = builder.getIndexType();
  auto i1 = builder.getIntegerType(1);
  auto none = builder.getNoneType();
  auto vt = hc::typing::ValueType::get(builder.getContext());
  if (checkCall("is_same", i1, {vt, vt})) {
    builder.replaceOpWithNewOp<hc::typing::IsSameOp>(call, callResType, args[0],
                                                     args[1]);
    return;
  }
  if (checkCall("check", none, {i1})) {
    builder.create<hc::typing::CheckOp>(call.getLoc(), args[0]);
    builder.replaceOpWithNewOp<hc::py_ir::NoneOp>(call);
    return;
  }
  if (checkCall("make_symbol", vt, {vt})) {
    builder.replaceOpWithNewOp<hc::typing::MakeSymbolOp>(call, callResType,
                                                         args[0]);
    return;
  }
  if (checkCall("get_num_args", index, {})) {
    builder.replaceOpWithNewOp<hc::typing::GetNumArgsOp>(call, callResType);
    return;
  }

  auto isStrLiteralArg = [&](llvm::StringRef funcName,
                             mlir::Type resType) -> mlir::StringAttr {
    if (callArgTypes.size() != 1)
      return nullptr;

    if (funcName != name)
      return nullptr;

    if (callResType != resType)
      return nullptr;

    auto literal =
        mlir::dyn_cast<hc::typing::LiteralType>(callArgTypes.front());
    if (!literal)
      return nullptr;

    return mlir::dyn_cast<mlir::StringAttr>(literal.getValue());
  };

  if (auto name = isStrLiteralArg("get_attr", vt)) {
    builder.replaceOpWithNewOp<hc::typing::GetAttrOp>(call, callResType, name);
    return;
  }
}

namespace {
struct GenResolversFuncsPass
    : public mlir::PassWrapper<GenResolversFuncsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenResolversFuncsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());
    auto visitor = [&](hc::py_ir::CallOp op) {
      auto funcType =
          mlir::dyn_cast<hc::typing::IdentType>(op.getFunc().getType());
      if (!funcType)
        return;

      llvm::StringRef funcName = funcType.getName().getValue();
      if (!funcName.consume_front("hckernel.typing."))
        return;

      convertCall(builder, op, funcName);
    };

    getOperation()->walk(visitor);
  }
};

struct DropFuncDecoratorPass
    : public mlir::PassWrapper<DropFuncDecoratorPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropFuncDecoratorPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto type =
        hc::typing::IdentType::get(&getContext(), "hckernel.typing.func");
    llvm::SmallVector<mlir::Value> decorators;
    auto visitor = [&](hc::py_ir::PyFuncOp op) {
      auto filter = [&](mlir::Value val) -> bool {
        return val.getType() != type;
      };

      decorators.clear();
      llvm::append_range(decorators,
                         llvm::make_filter_range(op.getDecorators(), filter));
      op.getDecoratorsMutable().assign(decorators);
    };

    getOperation()->walk<mlir::WalkOrder::PostOrder>(visitor);
  }
};
} // namespace

static std::optional<mlir::ArrayAttr>
processDecorators(mlir::ValueRange decorators) {
  if (decorators.size() != 1)
    return std::nullopt;

  auto type =
      mlir::dyn_cast<hc::typing::IdentType>(decorators.front().getType());
  if (!type || type.getName() != "type_resolver_type" ||
      type.getParamNames().size() != 1 || type.getParamNames().front() != "key")
    return std::nullopt;

  auto seqType =
      mlir::dyn_cast<hc::typing::SequenceType>(type.getParams().front());
  if (!seqType)
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute> ret;
  for (auto elem : seqType.getParams()) {
    auto lit = mlir::dyn_cast<hc::typing::LiteralType>(elem);
    if (!lit)
      return std::nullopt;

    ret.emplace_back(lit.getValue());
  }
  return mlir::ArrayAttr::get(type.getContext(), ret);
}

namespace {
struct GenResolversPass
    : public mlir::PassWrapper<GenResolversPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenResolversPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());
    auto visitor = [&](hc::py_ir::PyModuleOp mod) -> mlir::WalkResult {
      builder.setInsertionPoint(mod);
      for (mlir::Operation &nestedOp :
           llvm::make_early_inc_range(mod.getBody()->getOperations())) {
        auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(nestedOp);
        if (!func) {
          if (auto funcLike =
                  mlir::dyn_cast<mlir::FunctionOpInterface>(nestedOp))
            funcLike->moveBefore(mod);

          continue;
        }

        if (!func.getCaptureArgs().empty()) {
          func->emitOpError("Cannot convert function with captures");
          return mlir::WalkResult::interrupt();
        }

        if (!llvm::all_of(func.getBlockArgs(), [](auto arg) {
              return mlir::isa<hc::typing::ValueType>(arg.getType());
            })) {
          func->emitOpError("Invalid arg types");
          return mlir::WalkResult::interrupt();
        }

        auto decorator = processDecorators(func.getDecorators());
        if (!decorator) {
          func->emitOpError("Invalid decorator");
          return mlir::WalkResult::interrupt();
        }

        auto funcVisitor = [&](mlir::Operation *op) -> mlir::WalkResult {
          if (!mlir::isa<hc::typing::TypingInterpreterInterface>(op) &&
              !mlir::isa<hc::py_ir::ReturnOp, hc::py_ir::PyFuncOp>(op)) {
            op->emitOpError("unsupported typing op");
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        };
        if (func.walk(funcVisitor).wasInterrupted())
          return mlir::WalkResult::interrupt();

        mlir::Location loc = func.getLoc();
        auto resolver =
            builder.create<hc::typing::TypeResolverOp>(loc, *decorator);
        mlir::Region &dstReg = resolver.getBodyRegion();
        builder.inlineRegionBefore(func.getBodyRegion(), dstReg,
                                   dstReg.begin());

        mlir::Block &entry = dstReg.front();
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&entry);
        for (auto &&[i, arg] : llvm::enumerate(entry.getArguments())) {
          mlir::Value idx =
              builder.create<mlir::arith::ConstantIndexOp>(loc, i);
          mlir::Value newArg =
              builder.create<hc::typing::GetArgOp>(loc, arg.getType(), idx);
          arg.replaceAllUsesWith(newArg);
        }
        entry.eraseArguments(0, entry.getNumArguments());

        for (mlir::Block &block : dstReg) {
          auto term =
              mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
          if (!term)
            continue;

          builder.setInsertionPoint(term);
          builder.replaceOpWithNewOp<hc::typing::TypeResolverReturnOp>(
              term, term.getOperand());
        }
      }

      mod->erase();
      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PreOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

void populateTypingPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createPyTypeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<GenResolversFuncsPass>());
  pm.addPass(std::make_unique<DropFuncDecoratorPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(hc::createPyIRPromoteFuncsToStaticPass());
  pm.addPass(hc::createPyTypeInferencePass());
  pm.addPass(hc::createDropTypeResolversPass());
  pm.addPass(hc::createConverPyFuncToFuncPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<GenResolversPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
