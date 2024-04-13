// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPING_OPS
#define TYPING_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Typing_Dialect : Dialect {
  let name = "typing";
  let cppNamespace = "::hc::typing";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

//  let useDefaultAttributePrinterParser = 1;
//  let useDefaultTypePrinterParser = 1;
}

#endif // TYPING_OPS
