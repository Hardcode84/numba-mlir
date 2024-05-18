// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pybind11/pybind11.h>

#include <mlir/IR/MLIRContext.h>

struct Context {
  mlir::MLIRContext context;
};

pybind11::capsule createContext();
