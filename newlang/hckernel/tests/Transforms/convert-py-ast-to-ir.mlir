// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-convert-py-ast-to-ir-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
py_ast.module {
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK: %[[R:.*]] = py_ast.name "A"
//       CHECK: py_ir.return %[[R]] : !py_ast.node
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "A"
    py_ast.return %0
  }
}
