// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYIR_OPS
#define PYIR_OPS

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
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
  let useDefaultTypePrinterParser = 1;
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


def PyIR_UndefinedType : PyIR_Type<"Undefined", "undefined", [], "::mlir::Type">;

def PyIR_IdentType : PyIR_Type<"Ident", "ident", [], "::mlir::Type"> {
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

def PyIR_SubscriptType : PyIR_Type<"Subscript", "subscript", [], "::mlir::Type"> {
  let parameters = (ins
      "::mlir::Type":$value,
      "::mlir::Type":$slice);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$value,
      "::mlir::Type":$slice
    ), [{
      auto ctx = value.getContext();
      return $_get(ctx, value, slice);
    }]>
  ];

  let assemblyFormat = "$value `[` $slice `]`";
}

def PyIR_ConstType : PyIR_Type<"Const", "const", [], "::mlir::Type"> {
  let parameters = (ins "::mlir::Attribute":$value);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Attribute":$value
    ), [{
      auto ctx = value.getContext();
      return $_get(ctx, value);
    }]>
  ];

  let assemblyFormat = "$value";
}

def PyIR_NoneOp : PyIR_OpBase<"none", [Pure]> {
  let results = (outs NoneType:$result);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::NoneType::get($_builder.getContext()));
    }]>
  ];

  let assemblyFormat = "attr-dict";
}

def PyIR_ConstantOp : PyIR_OpBase<"constant",
    [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::TypedAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let assemblyFormat = "attr-dict $value";

  let hasFolder = 1;
}

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

def PyIR_PyFuncOp : PyIR_OpBase<"func", [Pure]> {

  let arguments = (ins
      StrAttr:$name,
      Variadic<AnyType>:$decorators);

  let results = (outs AnyType:$result);

  let regions = (region AnyRegion:$bodyRegion);

  let builders = [
    OpBuilder<(ins
        "::mlir::Type":$resultType,
        "::llvm::StringRef":$name,
        "::mlir::TypeRange":$argTypes,
        "::mlir::ValueRange":$decorators)>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = "attr-dict $name ( `decorators` $decorators^ `:` type($decorators) )? `->` type($result) $bodyRegion";
}

def PyIR_ReturnOp : PyIR_OpBase<"return", [Pure, HasParent<"PyFuncOp">,
                                      ReturnLike, Terminator]> {

  let arguments = (ins AnyType:$operand);

  let assemblyFormat = "attr-dict $operand `:` type($operand)";
}

def PyIR_StoreVarOp : PyIR_OpBase<"storevar"> {
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value);

  let assemblyFormat = "attr-dict $name $value `:` type($value)";
}

def PyIR_CallOp : PyIR_OpBase<"call"> {
  let arguments = (ins
    AnyType:$func,
    Variadic<AnyType>:$args,
    ArrayAttr:$args_names);

  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $func `:` type($func) ` ` custom<ArgList>($args, $args_names) (`:` type($args)^)? `->` type($result)";
}

def PyIR_LoadVarOp : PyIR_OpBase<"loadvar", [NoMemoryEffect]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $name `:` type($result)";
}

def PyIR_CastOp : PyIR_OpBase<"cast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def PyIR_SetItemOp : PyIR_OpBase<"setitem"> {
  let arguments = (ins
    AnyType:$target,
    AnyType:$index,
    AnyType:$value);

  let assemblyFormat = "attr-dict $target `:` type($target) `[` $index `:` type($index) `]` `=` $value `:` type($value)";
}

def PyIR_GetItemOp : PyIR_OpBase<"getitem"> {
  let arguments = (ins
    AnyType:$target,
    AnyType:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $target `:` type($target) `[` $index `:` type($index) `]` `->` type($result)";
}

def PyIR_SetAttrOp : PyIR_OpBase<"setattr"> {
  let arguments = (ins
    AnyType:$target,
    StrAttr:$name,
    AnyType:$value);

  let assemblyFormat = "attr-dict $target `:` type($target) `attr` $name `=` $value `:` type($value)";
}

def PyIR_GetAttrOp : PyIR_OpBase<"getattr"> {
  let arguments = (ins
    AnyType:$target,
    StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $target `:` type($target) `attr` $name `->` type($result)";
}

def PyIR_TuplePackOp : PyIR_OpBase<"tuple_pack"> {
  let arguments = (ins Variadic<AnyType>:$args);

  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $args `:` type($args) `->` type($result)";
}

def PyIR_TupleUnpackOp : PyIR_OpBase<"tuple_unpack"> {
  let arguments = (ins AnyType:$arg);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "attr-dict $arg `:` type($arg) `->` type($results)";
}

def PyIR_SliceOp : PyIR_OpBase<"slice", [AttrSizedOperandSegments]> {
  let arguments = (ins
      Optional<AnyType>:$lower,
      Optional<AnyType>:$upper,
      Optional<AnyType>:$step);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    attr-dict `(` $lower ` ` type($lower) `:`
                  $upper ` ` type($upper) `:`
                  $step  ` ` type($step)  `)`
                  `->` type($result)
  }];
}


def PyIR_BinOpAttr : I64EnumAttr<
    "BinOpVal", "",
    [
      I64EnumAttrCase<"add", 0>,
      I64EnumAttrCase<"sub", 1>,
      I64EnumAttrCase<"mul", 2>,
      I64EnumAttrCase<"div", 3>,
      I64EnumAttrCase<"floor_div", 4>,
      I64EnumAttrCase<"mod", 5>,
      I64EnumAttrCase<"pow", 6>,
      I64EnumAttrCase<"lshift", 7>,
      I64EnumAttrCase<"rshift", 8>,
      I64EnumAttrCase<"bit_or", 9>,
      I64EnumAttrCase<"bit_xor", 10>,
      I64EnumAttrCase<"bit_and", 11>,
      I64EnumAttrCase<"matmul", 12>,
    ]> {
  let cppNamespace = "::hc::py_ir";
}

def PyIR_BinOp : PyIR_OpBase<"binop"> {
  let arguments = (ins
      AnyType:$left,
      PyIR_BinOpAttr:$op,
      AnyType:$right);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $left`:`type($left) $op $right`:`type($right) `->` type($result)";
}

#endif // PYIR_OPS
