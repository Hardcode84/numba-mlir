// RUN: hc-opt -split-input-file %s --hc-py-type-inference-pass | FileCheck %s


typing.type_resolver ["py_ir.loadvar", "CurrentGroup"] {
  %0 = typing.make_ident "CurrentGroup" []
  typing.type_resolver_return %0
}

//       CHECK: ![[ID:.*]] = !typing<ident "CurrentGroup">
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[ID]]):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : ![[ID]] attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "CurrentGroup" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "Foo"] {
  %0 = typing.make_ident "Foo" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "Bar"] {
  %0 = typing.make_symbol "Bar"
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "Baz"] {
  %0 = typing.make_symbol "Baz"
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.tuple_pack"] {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = typing.get_num_args
  %seq = typing.create_seq
  cf.br ^bb1(%c0, %seq : index, !typing.res_val)
^bb1(%idx: index, %s1: !typing.res_val):
  %cond = arith.cmpi slt, %idx, %n : index
  %next = arith.addi %idx, %c1 : index
  cf.cond_br %cond, ^bb2, ^bb3(%s1: !typing.res_val)
^bb2:
  %arg = typing.get_arg %idx
  %new_seq = typing.append_seq %s1 %arg
  cf.br ^bb1(%next, %new_seq : index, !typing.res_val)
^bb3(%s2: !typing.res_val):
  %0 = typing.make_ident "Tuple" ["Elements"] : %s2
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.getitem"] {
  %c1 = arith.constant 1 : index
  %tuple = typing.get_arg %c1
  %shape = typing.get_ident_param %tuple "Elements"
  %0 = typing.make_ident "Foo" ["Elements"] : %shape
  typing.type_resolver_return %0
}

//       CHECK: ![[SYM1:.*]] = !typing<symbol "Bar">
//       CHECK: ![[SYM2:.*]] = !typing<symbol "Baz">
//       CHECK: ![[SEQ:.*]] = !typing<sequence ![[SYM1]], ![[SYM2]]>
//       CHECK: ![[ID:.*]] = !typing<ident "Foo" : "Elements" -> ![[SEQ]]>
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[ID]]):

py_ir.module {
  %0 = py_ir.loadvar "Bar" : !py_ir.undefined
  %1 = py_ir.loadvar "Baz" : !py_ir.undefined
  %2 = py_ir.tuple_pack %0, %1 : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
  %3 = py_ir.loadvar "Foo" : !py_ir.undefined
  %4 = py_ir.getitem %3 : !py_ir.undefined[%2 : !py_ir.undefined] -> !py_ir.undefined
  %5 = py_ir.func "func" (A:%4) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %7 = py_ir.none
    py_ir.return %7 : none
  }
  %6 = py_ir.call %5 : !py_ir.undefined  () -> !py_ir.undefined
}
