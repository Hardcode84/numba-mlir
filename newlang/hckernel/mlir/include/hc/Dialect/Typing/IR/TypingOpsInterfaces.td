// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPING_OPS_INTERFACES
#define TYPING_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def TypingKeyInterface : OpInterface<"TypingKeyInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getTypingKey", (ins)>,
  ];

}

#endif // TYPING_OPS_INTERFACES
