// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pybind11/pybind11.h>

#include <mlir/IR/MLIRContext.h>

struct Settings {
  bool dumpIR = false;
};

struct Context {
  mlir::MLIRContext context;
  Settings settings;
};

pybind11::capsule createContext(pybind11::dict settings);
