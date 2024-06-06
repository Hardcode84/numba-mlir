// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingDispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include "Context.hpp"
#include "DecoratorPipeline.hpp"
#include "MlirWrappers.hpp"

namespace py = pybind11;

class PyMlirWrapperBase : public MlirWrapperBase {
public:
  using MlirWrapperBase::MlirWrapperBase;

  void updatePipeline(std::string name, DecoratorPipeline *pipeline) override {
    PYBIND11_OVERRIDE_PURE(
        void,            /* Return type */
        MlirWrapperBase, /* Parent class */
        updatePipeline,  /* Name of function in C++ (must match Python name) */
        static_cast<void *>(pipeline) /* Argument(s) */
    );
  }
};

void addBuildinTypes(py::module_ &m) {
  auto *wrapper = new MlirTypeWrapper([](mlir::MLIRContext *context) {
    return mlir::IntegerType::get(context, 1);
  });
  m.add_object("i1", py::cast(wrapper));
}

hc::py_ir::PyFuncOp wrapOp(mlir::PatternRewriter &rewriter,
                           llvm::StringRef name,
                           mlir::ArrayRef<mlir::StringRef> argNames,
                           mlir::TypeRange argTypes,
                           const MlirOpWrapper::MakeOp &makeOp) {
  auto loc = rewriter.getUnknownLoc();
  auto emptyAnnotation = rewriter.create<hc::py_ir::EmptyAnnotationOp>(loc);
  mlir::SmallVector<mlir::Value> annotations(argNames.size(), emptyAnnotation);
  auto undef = hc::py_ir::UndefinedType::get(rewriter.getContext());

  auto func = rewriter.create<hc::py_ir::PyFuncOp>(
      loc, undef, name, argNames, mlir::ValueRange(annotations),
      mlir::ArrayRef<::llvm::StringRef>{}, mlir::ValueRange{},
      mlir::ValueRange{});

  auto &region = func.getBodyRegion();
  auto *block = &region.front();

  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    auto blockArgs = block->getArguments();

    mlir::SmallVector<mlir::Value> castedArgs;
    for (int i = 0; i < argNames.size(); ++i) {
      auto val = blockArgs[i];
      auto castResult =
          rewriter.create<hc::typing::CastOp>(loc, argTypes[i], val)
              .getResult();
      castedArgs.push_back(castResult);
    }
    auto results = makeOp(rewriter, loc, castedArgs);

    if (results.size() == 0) {
      auto none = rewriter.create<hc::py_ir::NoneOp>(loc);
      rewriter.create<hc::py_ir::ReturnOp>(loc, none);
    } else if (results.size() == 1) {
      rewriter.create<hc::py_ir::ReturnOp>(loc, results[0]);
    } else {
      auto tuple = rewriter.create<hc::py_ir::TuplePackOp>(loc, undef, results);
      rewriter.create<hc::py_ir::ReturnOp>(loc, tuple);
    }
  }

  auto attrName = mlir::StringAttr::get(rewriter.getContext(), "force_inline");
  auto unitAttr = mlir::UnitAttr::get(rewriter.getContext());

  func->setAttr(attrName, unitAttr);

  return func;
}

mlir::ValueRange makeCheckOp(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::ValueRange args) {
  rewriter.create<hc::typing::CheckOp>(loc, args[0]);

  return {};
}

mlir::Operation *makeWrappedCheckOp(mlir::PatternRewriter &rewriter) {
  auto i1 = rewriter.getI1Type();
  return wrapOp(rewriter, "check", {"test"}, {i1}, makeCheckOp);
}

mlir::ValueRange makeIsSameOp(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::ValueRange args) {
  auto i1 = rewriter.getI1Type();
  auto isSame =
      rewriter.create<hc::typing::IsSameOp>(loc, i1, args[0], args[1]);

  return isSame.getResult();
}

mlir::Operation *makeWrappedIsSameOp(mlir::PatternRewriter &rewriter) {
  auto valueType = hc::typing::ValueType::get(rewriter.getContext());
  return wrapOp(rewriter, "is_same", {"a", "b"}, {valueType, valueType},
                makeIsSameOp);
}

void addBuildinOps(py::module_ &m) {
  auto *check = new MlirOpWrapper(makeWrappedCheckOp);
  m.add_object("check", py::cast(check));

  auto *is_same = new MlirOpWrapper(makeWrappedIsSameOp);
  m.add_object("is_same", py::cast(is_same));
}

void addBuildinDecorators(py::object obj, py::capsule ctx) {
  // Mock
}

mlir::Operation *makeIdent(const MlirTypeWrapper::TypeConstructor &constr,
                           mlir::PatternRewriter &rewriter) {
  auto *context = rewriter.getContext();
  auto type = constr(context);
  std::string typeDescr;

  llvm::raw_string_ostream rs(typeDescr);
  type.print(rs);

  auto loc = rewriter.getUnknownLoc();

  auto ident = rewriter.create<hc::typing::MakeIdentOp>(
      loc, hc::typing::ValueType::get(context),
      rewriter.getStringAttr(typeDescr), rewriter.getArrayAttr({}),
      mlir::ValueRange());

  return ident;
}

MlirTypeWrapper::MlirTypeWrapper(const MlirTypeWrapper::TypeConstructor &constr)
    : MlirOpWrapper([constr](mlir::PatternRewriter &rewriter) {
        return makeIdent(constr, rewriter);
      }) {
  this->typeConstr = constr;
}

void MlirTypeWrapper::definePyClass(pybind11::module_ &m) {
  py::class_<MlirTypeWrapper, MlirOpWrapper>(m, "MlirTypeWrapper")
      .def(py::init<>());
}

MlirOpWrapper::MlirOpWrapper(const hc::OpConstructorFunc &op) {
  this->opConstr = op;
}

void MlirOpWrapper::updatePipeline(std::string name,
                                   DecoratorPipeline *pipeline) {
  pipeline->addVariable(name, this);
}

void MlirOpWrapper::definePyClass(pybind11::module_ &m) {
  py::class_<MlirOpWrapper, MlirWrapperBase>(m, "MlirOpWrapper")
      .def(py::init<>());
}

void MlirWrapperBase::definePyClass(pybind11::module_ &m) {
  py::class_<MlirWrapperBase, PyMlirWrapperBase>(m, "MlirWrapperBase")
      .def(py::init<>());
}

MlirDecoratorWrapper::MlirDecoratorWrapper(std::string name) {
  this->name = name;
}

void MlirDecoratorWrapper::definePyClass(pybind11::module_ &m) {
  py::class_<MlirDecoratorWrapper, MlirWrapperBase>(m, "MlirDecoratorWrapper")
      .def(py::init<>());
}

void MlirDecoratorWrapper::updatePipeline(std::string name,
                                          DecoratorPipeline *pipeline) {
  pipeline->addDecorator(name, this);
}
