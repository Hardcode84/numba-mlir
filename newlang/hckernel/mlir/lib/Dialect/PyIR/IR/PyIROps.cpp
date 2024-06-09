// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::py_ir::PyIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"
      >();
}

void hc::py_ir::LoadModuleOp::getTypingKeyArgs(
    llvm::SmallVectorImpl<mlir::Attribute> &args) {
  args.emplace_back(getNameAttr());
}

namespace {

struct CleanupUnusedCaptures final
    : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange capturesArgs = op.getCaptureArgs();
    mlir::ValueRange capturesBlockArgs = op.getCaptureBlockArgs();
    mlir::SmallVector<unsigned> indexes;
    mlir::SmallVector<mlir::Operation *> constOps;
    for (int i = capturesBlockArgs.size() - 1; i >= 0; --i) {
      if (capturesBlockArgs[i].use_empty()) {
        indexes.emplace_back(i);
        constOps.emplace_back(nullptr);
        continue;
      }

      mlir::Operation *def = capturesArgs[i].getDefiningOp();
      if (def && def->getNumOperands() == 0 && def->getNumResults() == 1 &&
          def->hasTrait<mlir::OpTrait::ConstantLike>()) {
        indexes.emplace_back(i);
        constOps.emplace_back(def);
        continue;
      }
    }

    if (indexes.empty())
      return mlir::failure();

    auto *entryBlock = op.getEntryBlock();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    rewriter.modifyOpInPlace(op, [&] {
      auto allArgs = entryBlock->getArguments().size();
      auto captNames =
          llvm::to_vector(op.getCaptureNames().getAsRange<mlir::Attribute>());

      unsigned numArgs = op.getBlockArgs().size();
      mlir::BitVector toErase(allArgs);
      for (auto &&[i, idx] : llvm::enumerate(indexes)) {
        int blockArgIdx = idx + numArgs;
        if (auto def = constOps[i]) {
          mlir::Operation *newOp = rewriter.clone(*def);
          rewriter.replaceAllUsesWith(entryBlock->getArgument(blockArgIdx),
                                      newOp->getResult(0));
        }
        op.getCaptureArgsMutable().erase(idx);
        toErase.set(blockArgIdx);
        captNames.erase(captNames.begin() + idx);
      }

      auto attr =
          rewriter.getArrayAttr(mlir::ArrayRef<mlir::Attribute>(captNames));
      op.setCaptureNamesAttr(attr);
      entryBlock->eraseArguments(toErase);
    });

    return mlir::success();
  }
};

} // namespace

mlir::OpFoldResult hc::py_ir::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

void hc::py_ir::PyFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Type resultType, llvm::StringRef name,
                                llvm::ArrayRef<llvm::StringRef> argNames,
                                mlir::ValueRange annotations,
                                llvm::ArrayRef<llvm::StringRef> captureNames,
                                mlir::ValueRange captureArgs,
                                mlir::ValueRange decorators) {
  assert(argNames.size() == annotations.size());
  assert(captureNames.size() == captureArgs.size());
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute(getArgNamesAttrName(odsState.name),
                        odsBuilder.getStrArrayAttr(argNames));
  odsState.addAttribute(getCaptureNamesAttrName(odsState.name),
                        odsBuilder.getStrArrayAttr(captureNames));
  odsState.addOperands(annotations);
  odsState.addOperands(captureArgs);
  odsState.addOperands(decorators);
  odsState.addTypes(resultType);

  int32_t segmentSizes[3] = {};
  segmentSizes[0] = static_cast<int32_t>(annotations.size());
  segmentSizes[1] = static_cast<int32_t>(captureArgs.size());
  segmentSizes[2] = static_cast<int32_t>(decorators.size());
  odsState.addAttribute(getOperandSegmentSizeAttr(),
                        odsBuilder.getDenseI32ArrayAttr(segmentSizes));

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);

  auto numArgs = argNames.size() + captureNames.size();
  llvm::SmallVector<mlir::Type> types(
      numArgs, UndefinedType::get(odsBuilder.getContext()));
  llvm::SmallVector<mlir::Location> locs(numArgs, odsBuilder.getUnknownLoc());
  odsBuilder.createBlock(region, {}, types, locs);
}

void hc::py_ir::PyFuncOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<CleanupUnusedCaptures>(context);
}

mlir::FailureOr<bool>
hc::py_ir::ConstantOp::inferTypes(mlir::TypeRange types,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  if (!types.empty())
    return emitError("Invalid arg count");

  results.emplace_back(hc::typing::LiteralType::get(this->getValue()));
  return true;
}

void hc::py_ir::LoadVarOp::getTypingKeyArgs(
    llvm::SmallVectorImpl<mlir::Attribute> &args) {
  args.emplace_back(getNameAttr());
}

static bool parseArgList(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &argsOperands,
    mlir::ArrayAttr &args_namesAttr) {
  if (parser.parseLParen())
    return true;

  auto *context = parser.getContext();
  llvm::SmallVector<mlir::Attribute> names;
  if (parser.parseOptionalRParen()) {
    std::string name;
    while (true) {
      name.clear();
      if (!parser.parseOptionalKeywordOrString(&name)) {
        if (parser.parseColon())
          return true;
      }
      names.push_back(mlir::StringAttr::get(context, name));

      argsOperands.push_back({});
      if (parser.parseOperand(argsOperands.back()))
        return true;

      if (!parser.parseOptionalRParen())
        break;

      if (parser.parseComma())
        return true;
    }
  }

  assert(names.size() == argsOperands.size());
  args_namesAttr = mlir::ArrayAttr::get(context, names);
  return false;
}

template <typename Op>
static void printArgList(mlir::OpAsmPrinter &printer, Op /*op*/,
                         mlir::ValueRange args, mlir::ArrayAttr argsNames) {
  assert(args.size() == argsNames.size());
  printer << '(';
  bool first = true;
  for (auto &&[arg, name] : llvm::zip(args, argsNames)) {
    if (first) {
      first = false;
    } else {
      printer << ", ";
    }
    auto nameStr =
        (name ? name.cast<mlir::StringAttr>().getValue() : llvm::StringRef());
    if (!nameStr.empty())
      printer << nameStr << ':';
    printer.printOperand(arg);
  }
  printer << ')';
}

#include "hc/Dialect/PyIR/IR/PyIROpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"

#include "hc/Dialect/PyIR/IR/PyIROpsEnums.cpp.inc"
