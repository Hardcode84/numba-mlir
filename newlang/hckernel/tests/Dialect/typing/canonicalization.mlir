// RUN: hc-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK: ![[EXPR:.*]] = !typing<expr 0>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr 0>)

// -----

// CHECK: ![[SYM:.*]] = !typing<symbol "A">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM]]) -> d0>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<symbol "A"> ) -> d0 >)

// -----

// CHECK: ![[SYM1:.*]] = !typing<symbol "A">
// CHECK: ![[SYM2:.*]] = !typing<symbol "B">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM1]], ![[SYM2]]) -> d0 + d1>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<symbol "A">, !typing<symbol "B"> ) -> d0 + d1 >)
