// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPING_OPS_INTERFACES
#define TYPING_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def TypingKeyInterface : OpInterface<"TypingKeyInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::Attribute",
      "getTypingKey",
      (ins),
      [{}],
      [{
        ::llvm::SmallVector<::mlir::Attribute> attrs;
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        attrs.emplace_back(op->getName().getIdentifier());
        op.getTypingKeyArgs(attrs);
        return ::mlir::ArrayAttr::get(op->getContext(), attrs);
      }]
      >,
    InterfaceMethod<
      "",
      "void",
      "getTypingKeyArgs",
      (ins "::llvm::SmallVectorImpl<::mlir::Attribute>&":$args)>
  ];

}

def TypingInterpreterInterface : OpInterface<"TypingInterpreterInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::LogicalResult",
      "interpret",
      (ins "::llvm::DenseMap<::mlir::Value, ::mlir::Type>&":$state)>
  ];

}


#endif // TYPING_OPS_INTERFACES
