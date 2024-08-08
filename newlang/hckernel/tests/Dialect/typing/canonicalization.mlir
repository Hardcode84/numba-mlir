// RUN: hc-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK: ![[EXPR:.*]] = !typing<expr 0>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr 0>)
