// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "DispatcherBase.hpp"

class Dispatcher : public DispatcherBase {
public:
  static void definePyClass(pybind11::module_ &m);

  using DispatcherBase::DispatcherBase;

  void call(pybind11::args args, pybind11::kwargs kwargs);
};
