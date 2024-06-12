typing.type_resolver ["py_ir.load_module", "hckernel"] {
  %0 = typing.make_ident "hckernel" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.getattr", "typing"] {
  %c1 = arith.constant 0: index
  %0 = typing.make_ident "hckernel" []
  %1 = typing.get_arg %c1
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing" []
  typing.type_resolver_return %3
}
