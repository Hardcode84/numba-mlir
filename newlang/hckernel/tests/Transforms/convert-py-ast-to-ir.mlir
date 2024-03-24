// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-convert-py-ast-to-ir-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
py_ast.module {
}
