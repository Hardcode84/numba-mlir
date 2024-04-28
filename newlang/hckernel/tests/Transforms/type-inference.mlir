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

typing.type_resolver ["py_ir.cast"] {
  %c0 = arith.constant 0 : index
  %0 = typing.get_arg %c0
  typing.type_resolver_return %0
}

//       CHECK: ![[LIT:.*]] = !typing<literal 1 : i64>
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[LIT]]):

py_ir.module {
  %0 = py_ir.constant 1 : i64
  %1 = py_ir.cast %0 : i64 to !py_ir.undefined
  %2 = py_ir.func "func" (A:%1) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %4 = py_ir.none
    py_ir.return %4 : none
  }
  %3 = py_ir.call %2 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.cast"] {
  %c0 = arith.constant 0 : index
  %0 = typing.get_arg %c0
  %1 = typing.make_literal 1 : i64
  %2 = typing.is_same %0 %1
  typing.check %2
  %3 = typing.make_ident "Foo" []
  typing.type_resolver_return %3
}

typing.type_resolver ["py_ir.cast"] {
  %c0 = arith.constant 0 : index
  %0 = typing.get_arg %c0
  %1 = typing.make_literal 2 : i64
  %2 = typing.is_same %0 %1
  typing.check %2
  %3 = typing.make_ident "Bar" []
  typing.type_resolver_return %3
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "Foo">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "Bar">
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%{{.*}}: ![[ID1]], %{{.*}}: ![[ID2]]):

py_ir.module {
  %0 = py_ir.constant 1 : i64
  %1 = py_ir.constant 2 : i64
  %2 = py_ir.cast %0 : i64 to !py_ir.undefined
  %3 = py_ir.cast %1 : i64 to !py_ir.undefined
  %4 = py_ir.func "func" (A:%2, A:%3) : !py_ir.undefined, !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined):
    %5 = py_ir.none
    py_ir.return %5 : none
  }
  %6 = py_ir.call %4 : !py_ir.undefined  () -> !py_ir.undefined
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

// -----

typing.type_resolver ["py_ir.loadvar", "B"] {
  %0 = typing.make_ident "B" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "C"] {
  %0 = typing.make_ident "C" []
  typing.type_resolver_return %0
}

typing.type_resolver "join_types" {
  %0 = typing.make_ident "D" []
  typing.type_resolver_return %0
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "B">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "C">
//   CHECK-DAG: ![[ID3:.*]] = !typing<ident "D">
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%{{.*}}: !py_ir.undefined, %[[B:.*]]: ![[ID1]], %[[C:.*]]: ![[ID2]]):
//       CHECK:  cf.cond_br %{{.*}}, ^bb1(%[[B]] : ![[ID1]]), ^bb2(%[[C]] : ![[ID2]])
//       CHECK:  ^bb1(%[[C1:.*]]: ![[ID1]]):
//       CHECK:  %[[C2:.*]] = py_ir.cast %[[C1]] : ![[ID1]] to ![[ID3]]
//       CHECK:  cf.br ^bb3(%[[C2]] : ![[ID3]])
//       CHECK:  ^bb2(%[[B1:.*]]: ![[ID2]]):
//       CHECK:  %[[B2:.*]] = py_ir.cast %[[B1]] : ![[ID2]] to ![[ID3]]
//       CHECK:  cf.br ^bb3(%[[B2]] : ![[ID3]])
//       CHECK:  ^bb3(%[[RES:.*]]: ![[ID3]]):
//       CHECK:  py_ir.return %[[RES]] : ![[ID3]]

py_ir.module {
  %0 = py_ir.loadvar "A" : !py_ir.undefined
  %2 = py_ir.loadvar "B" : !py_ir.undefined
  %3 = py_ir.loadvar "C" : !py_ir.undefined
  %4 = py_ir.func "func" () capture (A:%0, B:%2, C:%3) : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined):
    %6 = py_ir.cast %arg0 : !py_ir.undefined to i1
    cf.cond_br %6, ^bb1(%arg2 : !py_ir.undefined), ^bb2(%arg3 : !py_ir.undefined)
  ^bb1(%8: !py_ir.undefined):
    cf.br ^bb3(%8 : !py_ir.undefined)
  ^bb2(%10: !py_ir.undefined):
    cf.br ^bb3(%10 : !py_ir.undefined)
  ^bb3(%11: !py_ir.undefined):
    py_ir.return %11 : !py_ir.undefined
  }
  %5 = py_ir.call %4 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "B"] {
  %0 = typing.make_ident "B" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "C"] {
  %0 = typing.make_ident "C" []
  typing.type_resolver_return %0
}

typing.type_resolver "join_types" {
  %0 = typing.make_ident "D" []
  typing.type_resolver_return %0
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "B">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "C">
//   CHECK-DAG: ![[ID3:.*]] = !typing<ident "D">
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%{{.*}}: !py_ir.undefined, %[[ARG1:.*]]: ![[ID1]], %[[ARG2:.*]]: ![[ID2]]):
//       CHECK:  %[[RES:.*]] = typing.resolve %{{.*}}, %[[ARG1]], %[[ARG2]] : i1, ![[ID1]], ![[ID2]] -> ![[ID3]] {
//       CHECK:  %[[R:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !py_ir.undefined
//       CHECK:  typing.resolve_yield %[[R]] : !py_ir.undefined
//       CHECK:  py_ir.return %[[RES]] : ![[ID3]]

py_ir.module {
  %0 = py_ir.loadvar "A" : !py_ir.undefined
  %1 = py_ir.loadvar "B" : !py_ir.undefined
  %2 = py_ir.loadvar "C" : !py_ir.undefined
  %3 = py_ir.func "func" () capture (A:%0, B:%1, C:%2) : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined):
    %5 = py_ir.cast %arg0 : !py_ir.undefined to i1
    %6 = arith.select %5, %arg1, %arg2 : !py_ir.undefined
    py_ir.return %6 : !py_ir.undefined
  }
  %4 = py_ir.call %3 : !py_ir.undefined  () -> !py_ir.undefined
}
