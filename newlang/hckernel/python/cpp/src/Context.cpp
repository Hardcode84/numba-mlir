// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Context.hpp"

pybind11::capsule createContext() {
  auto ctx = std::make_unique<Context>();
  auto dtor = [](void *ptr) { delete static_cast<Context *>(ptr); };
  pybind11::capsule ret(ctx.get(), dtor);
  ctx.release();
  return ret;
}
