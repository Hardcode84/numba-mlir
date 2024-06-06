// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>

#include <pybind11/pybind11.h>

struct Context;

class TypingDispatcher {
public:
  static void definePyClass(pybind11::module_ &m);

  TypingDispatcher(pybind11::capsule ctx, pybind11::object getDesc,
                   pybind11::dict globals);
  ~TypingDispatcher();

  void compile();

private:
  Context &context;
  pybind11::object contextRef; // to keep context alive
  pybind11::object getFuncDesc;
  pybind11::dict globals;

  mlir::OwningOpRef<mlir::Operation *> mod;
};
