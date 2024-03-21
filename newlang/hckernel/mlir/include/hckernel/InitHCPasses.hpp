// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hckernel/Transforms/Passes.hpp"

namespace hckernel {
inline void registerAllPasses() { registerTransformsPasses(); }
} // namespace hckernel
