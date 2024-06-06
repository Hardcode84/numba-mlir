// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>

#include "hc/Utils.hpp"
#include <pybind11/pybind11.h>

#include <functional>

namespace mlir {
class Location;
class BlockArgument;
class ValueRange;
class PatternRewriter;
} // namespace mlir

class DecoratorPipeline;

class MlirWrapperBase {
public:
  static void definePyClass(pybind11::module_ &m);

  MlirWrapperBase() = default;
  virtual ~MlirWrapperBase() = default;

  virtual void updatePipeline(std::string name,
                              DecoratorPipeline *pipeline) = 0;
};

class MlirOpWrapper : public MlirWrapperBase {
public:
  using MakeOp = std::function<mlir::ValueRange(
      mlir::PatternRewriter &, mlir::Location, mlir::ValueRange)>;
  static void definePyClass(pybind11::module_ &m);

  MlirOpWrapper(const hc::OpConstructorFunc &op);
  MlirOpWrapper() = default;
  ~MlirOpWrapper() = default;

  void updatePipeline(std::string name, DecoratorPipeline *pipeline) override;

  const hc::OpConstructorFunc &getOpConstrutor() { return opConstr; }

protected:
  hc::OpConstructorFunc opConstr;
};

class MlirTypeWrapper : public MlirOpWrapper {
public:
  using TypeConstructor = std::function<mlir::Type(mlir::MLIRContext *)>;

  static void definePyClass(pybind11::module_ &m);

  MlirTypeWrapper(const TypeConstructor &typeConstr);
  MlirTypeWrapper() = default;
  ~MlirTypeWrapper() = default;

protected:
  TypeConstructor typeConstr;
};

class MlirDecoratorWrapper : public MlirWrapperBase {
public:
  static void definePyClass(pybind11::module_ &m);

  MlirDecoratorWrapper(std::string name);
  MlirDecoratorWrapper() = default;
  ~MlirDecoratorWrapper() = default;

  void updatePipeline(std::string name, DecoratorPipeline *pipeline) override;

protected:
  std::string name;
};

void addBuildinTypes(pybind11::module_ &);
void addBuildinOps(pybind11::module_ &);
void addBuildinDecorators(pybind11::object types, pybind11::capsule ctx);
