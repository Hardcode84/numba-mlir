// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-convert-py-ast-to-ir-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
py_ast.module {
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "A"
    py_ast.return %0
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "b" %[[ARG2]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "c" %[[ARG3]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.arg "a"
  %1 = py_ast.arg "b"
  %2 = py_ast.arg "c"
  py_ast.func "func"(%0, %1, %2) {
    %3 = py_ast.name "A"
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir<ident "Foo">):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir<ident "Foo">
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "Foo"
  %1 = py_ast.arg "a" : %0
  py_ast.func "func"(%1) {
    %3 = py_ast.name "A"
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1:.*]] = py_ir.loadvar "Cond" : !py_ir.undefined
//       CHECK:  %[[C2:.*]] = py_ir.cast %[[C1]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[C2]], ^bb1, ^bb2
//       CHECK:  ^bb1
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  cf.br ^bb2
//       CHECK:  ^bb2
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "Cond"
    py_ast.if %0 {
      "test.test1"() : () -> ()
    } {}
    %1 = py_ast.name "A"
    py_ast.return %1
  }
}


// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1:.*]] = py_ir.loadvar "Cond" : !py_ir.undefined
//       CHECK:  %[[C2:.*]] = py_ir.cast %[[C1]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[C2]], ^bb1, ^bb2
//       CHECK:  ^bb1
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  cf.br ^bb3
//       CHECK:  ^bb2
//       CHECK:  "test.test2"() : () -> ()
//       CHECK:  cf.br ^bb3
//       CHECK:  ^bb3
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "Cond"
    py_ast.if %0 {
      "test.test1"() : () -> ()
    } {
      "test.test2"() : () -> ()
    }
    %1 = py_ast.name "A"
    py_ast.return %1
  }
}
