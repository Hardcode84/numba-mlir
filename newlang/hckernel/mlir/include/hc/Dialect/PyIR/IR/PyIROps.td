// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYIR_OPS
#define PYIR_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def PyIR_Dialect : Dialect {
  let name = "py_ir";
  let cppNamespace = "::hc::py_ir";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

//  let useDefaultAttributePrinterParser = 1;
//  let useDefaultTypePrinterParser = 1;
}

class PyIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<PyIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class PyIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<PyIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class PyIR_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<PyIR_Dialect, mnemonic, traits>;


def PyIR_PyModuleOp : PyIR_OpBase<"module", [
      IsolatedFromAbove, SingleBlockImplicitTerminator<"PyModuleEndOp">
    ]> {


  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

def PyIR_PyModuleEndOp : PyIR_OpBase<"module_end", [
  Terminator, HasParent<"PyModuleOp">
]> {
  let assemblyFormat = "attr-dict";
}

#endif // PYIR_OPS
