// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Context.hpp"

namespace py = pybind11;

static void readSettings(Settings &ret, py::dict &dict) {
  ret.dumpIR = dict["DUMP_IR"].cast<bool>();
}

py::capsule createContext(py::dict settings) {
  auto ctx = std::make_unique<Context>();
  readSettings(ctx->settings, settings);
  auto dtor = [](void *ptr) { delete static_cast<Context *>(ptr); };
  pybind11::capsule ret(ctx.get(), dtor);
  ctx.release();
  return ret;
}
