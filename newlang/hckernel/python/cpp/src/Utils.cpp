// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Utils.hpp"

#include <stdexcept>

#include <llvm/ADT/Twine.h>

[[noreturn]] void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}
