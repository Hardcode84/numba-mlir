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

class Typing_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<Typing_Dialect, mnemonic, traits>;

def Typing_ResolveAnnotationOp : Typing_OpBase<"resolve_annotation", [Pure]> {
  let arguments = (ins AnyType:$value, AnyType:$annotation);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    attr-dict $value `:` type($value)
    `annotation` $annotation `:` type($annotation)
    `->` type($result)
  }];
}

def Typing_ResolveOp : Typing_OpBase<"resolve", [
      SingleBlockImplicitTerminator<"ResolveYieldOp">, RecursiveMemoryEffects
    ]> {

  let arguments = (ins Variadic<AnyType>:$args);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "attr-dict-with-keyword ($args^ `:` type($args))? $bodyRegion";
}

def Typing_ResolveYieldOp : Typing_OpBase<"resolve_yield", [
  Terminator, HasParent<"ResolveOp ">, Pure
]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($args^ `:` type($args))?";
}


#endif // TYPING_OPS
