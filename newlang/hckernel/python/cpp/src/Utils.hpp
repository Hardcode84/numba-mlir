// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace llvm {
class Twine;
}

[[noreturn]] void reportError(const llvm::Twine &msg);
