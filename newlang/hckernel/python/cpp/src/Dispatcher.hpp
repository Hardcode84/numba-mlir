// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/OwningOpRef.h>

#include <pybind11/pybind11.h>

struct Context;

class Dispatcher {
public:
  static void definePyClass(pybind11::module_ &m);

  Dispatcher(pybind11::capsule ctx, pybind11::object getDesc);
  ~Dispatcher();

  void call(pybind11::args args, pybind11::kwargs kwargs);

private:
  Context &context;
  pybind11::object contextRef; // to keep context alive
  pybind11::object getFuncDesc;
  mlir::OwningOpRef<mlir::Operation *> mod;

  struct ExceptionDesc {
    std::string message;
  };

  using FuncT = int (*)(ExceptionDesc *exc, PyObject *args, PyObject *kwargs);
  FuncT func = nullptr;
};
