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
  let useDefaultTypePrinterParser = 1;
}

class Typing_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Typing_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class Typing_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<Typing_Dialect, mnemonic, traits>;


def Typing_IdentType : Typing_Type<"Ident", "ident", [], "::mlir::Type"> {
  let parameters = (ins
    "::mlir::StringAttr":$name,
    ArrayRefParameter<"::mlir::StringAttr">:$paramNames,
    ArrayRefParameter<"::mlir::Type">:$params
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::MLIRContext*":$ctx,
      "::llvm::StringRef":$name,
      "::llvm::ArrayRef<::llvm::StringRef>":$paramNames,
      "::llvm::ArrayRef<::mlir::Type>":$params
    ), [{
      ::llvm::SmallVector<::mlir::StringAttr> arr;
      for (auto str : paramNames)
        arr.emplace_back(::mlir::StringAttr::get(ctx, str));
      return $_get(ctx, ::mlir::StringAttr::get(ctx, name), arr, params);
    }]>
  ];

  let assemblyFormat = "$name $paramNames `:` $params";
}

def Typing_SequenceType : Typing_Type<"Sequence", "sequence", [], "::mlir::Type"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$params
  );

  let assemblyFormat = "$params";
}

def Typing_SymbolType : Typing_Type<"Symbol", "symbol", [], "::mlir::Type"> {
  let parameters = (ins "::mlir::StringAttr":$name);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::MLIRContext*":$ctx,
      "::llvm::StringRef":$name
    ), [{
      return $_get(ctx, ::mlir::StringAttr::get(ctx, name));
    }]>
  ];

  let assemblyFormat = "$name";
}

def Typing_LiteralType : Typing_Type<"Literal", "literal", [], "::mlir::Type"> {
  let parameters = (ins "::mlir::TypedAttr":$value);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::TypedAttr":$value
    ), [{
      return $_get(value.getContext(), value);
    }]>
  ];

  let assemblyFormat = "$value";
}


def Typing_ResolveOp : Typing_OpBase<"resolve", [
      SingleBlockImplicitTerminator<"ResolveYieldOp">, RecursiveMemoryEffects
    ]> {

  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins
        "::mlir::TypeRange":$resultTypes,
        "::mlir::ValueRange":$args
      )>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    attr-dict-with-keyword ($args^ `:` type($args))?
    (`->` type($results)^)?
    $bodyRegion
  }];
}

def Typing_ResolveYieldOp : Typing_OpBase<"resolve_yield", [
  Terminator, HasParent<"ResolveOp ">, Pure
]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($args^ `:` type($args))?";
}


#endif // TYPING_OPS
