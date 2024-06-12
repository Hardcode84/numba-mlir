typing.type_resolver ["py_ir.load_module", "hckernel"] {
  %0 = typing.make_ident "hckernel" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.getattr", "typing"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.getattr", "type_resolver"] {
  %c1 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing" []
  %1 = typing.get_arg %c1
  %2 = typing.is_same %0 %1
  typing.check %2

  %3 = typing.make_ident "hckernel.typing.type_resolver" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.make_list"] {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = typing.get_num_args
  %seq = typing.create_seq
  cf.br ^bb1(%c0, %seq : index, !typing.value)
^bb1(%idx: index, %s1: !typing.value):
  %cond = arith.cmpi slt, %idx, %n : index
  %next = arith.addi %idx, %c1 : index
  cf.cond_br %cond, ^bb2, ^bb3(%s1: !typing.value)
^bb2:
  %arg = typing.get_arg %idx
  %new_seq = typing.append_seq %s1 %arg
  cf.br ^bb1(%next, %new_seq : index, !typing.value)
^bb3(%s2: !typing.value):
  %0 = typing.make_ident "List" ["Elements"] : %s2
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.call"] {
  %c0 = arith.constant 0: index
  %0 = typing.make_ident "hckernel.typing.type_resolver" []
  %1 = typing.get_arg %c0
  %2 = typing.is_same %0 %1
  typing.check %2

  %c1 = arith.constant 1: index
  %3 = typing.get_arg %c1
  %4 = typing.get_ident_name %3
  %5 = typing.make_literal "List"
  %6 = typing.is_same %4 %5
  typing.check %6

  %7 = typing.get_ident_param %3 "Elements"
  %8 = typing.make_ident "type_resolver_type" ["key"] : %7
  typing.type_resolver_return %8
}
