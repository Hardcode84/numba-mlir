// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-convert-py-ast-to-ir-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
py_ast.module {
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ast.name "A"
//       CHECK:  py_ir.return %[[R]] : !py_ast.node
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
//       CHECK:  %[[R:.*]] = py_ast.name "A"
//       CHECK:  py_ir.return %[[R]] : !py_ast.node
py_ast.module {
  %0 = py_ast.arg "a"
  %1 = py_ast.arg "b"
  %2 = py_ast.arg "c"
  py_ast.func "func"(%0, %1, %2) {
    %3 = py_ast.name "A"
    py_ast.return %3
  }
}
