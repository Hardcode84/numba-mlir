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
        auto op = ::mlir::cast<ConcreteOp>(this->getOperation());
        attrs.emplace_back(op->getName().getIdentifier());
        op.getTypingKeyArgs(attrs);
        return ::mlir::ArrayAttr::get(op->getContext(), attrs);
      }]
      >,
    InterfaceMethod<
      "",
      "void",
      "getTypingKeyArgs",
      (ins "::llvm::SmallVectorImpl<::mlir::Attribute>&":$args),
      [{}],
      [{
        // Nothing
      }]
      >
  ];
}

def TypingInterpreterInterface : OpInterface<"TypingInterpreterInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::FailureOr<bool>",
      "interpret",
      (ins "::hc::typing::InterpreterState&":$state)>
  ];
}

def TypeInferenceInterface : OpInterface<"TypeInferenceInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::FailureOr<bool>",
      "inferTypes",
      (ins
        "::mlir::TypeRange":$types,
        "::llvm::SmallVectorImpl<::mlir::Type>&":$results
      )>
  ];
}

def TypingUpdateInplaceInterface : OpInterface<"TypingUpdateInplaceInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "bool",
      "canUpdateArgTypeInplace",
      (ins "::mlir::Value":$arg),
      [{}],
      [{
        return true;
      }]
      >
  ];
}

#endif // TYPING_OPS_INTERFACES
