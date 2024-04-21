// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

using State = llvm::DenseMap<mlir::Value, mlir::Type>;

template <typename Dst, typename Src>
static auto castArrayRef(mlir::ArrayRef<Src> src) {
  return mlir::ArrayRef<Dst>(static_cast<const Dst *>(src.data()), src.size());
}

static mlir::Type getType(const State &state, mlir::Value val) {
  auto it = state.find(val);
  assert(it != state.end());
  return it->second;
}

static llvm::SmallVector<mlir::Type> getTypes(const State &state,
                                              mlir::ValueRange vals) {
  llvm::SmallVector<mlir::Type> ret(vals.size());
  for (auto &&[i, val] : llvm::enumerate(vals))
    ret[i] = getType(state, val);

  return ret;
}

static bool handleMakeIdent(State &state, hc::typing::MakeIdent op) {
  auto name = op.getNameAttr();
  auto paramNames =
      castArrayRef<mlir::StringAttr>(op.getParamNames().getValue());
  auto paramTypes = getTypes(state, op.getParams());
  state[op.getResult()] =
      hc::typing::IdentType::get(op.getContext(), name, paramNames, paramTypes);
  return true;
}

static bool handleOp(State &state, mlir::Operation &op) {
  if (auto makeIdent = mlir::dyn_cast<hc::typing::MakeIdent>(op)) {
    return handleMakeIdent(state, makeIdent);
  }
  return false;
}

mlir::FailureOr<llvm::SmallVector<mlir::Type>>
hc::typing::Interpreter::run(TypeResolverOp resolver, mlir::TypeRange types) {
  state.clear();
  assert(!resolver.getBodyRegion().empty());
  mlir::Block *block = &resolver.getBodyRegion().front();
  while (true) {
    for (mlir::Operation &op : block->without_terminator()) {
      if (!handleOp(state, op))
        return op.emitError("Type interpreter: unsupported op");

      auto term = block->getTerminator();
      if (auto ret = mlir::dyn_cast<TypeResolverReturnOp>(term)) {
        return getTypes(state, ret.getArgs());
      } else {
        return term->emitError("Unsupported terminator");
      }
    }
  }
  llvm_unreachable("Unreachable");
}
