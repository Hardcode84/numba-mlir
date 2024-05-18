// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pybind11/pybind11.h>

class Dispatcher {
public:
  Dispatcher(pybind11::object getSrc);

  void call(pybind11::args args, pybind11::kwargs kwargs);

private:
  pybind11::object getSourceFunc;
};
