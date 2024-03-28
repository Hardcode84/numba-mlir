// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-reconstruct-py-ssa-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.return %[[B]] : none
py_ir.module {
  %f = py_ir.func "foo" -> !py_ir.undefined {
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  py_ir.return %[[B1]] : none
py_ir.module {
  %f = py_ir.func "foo" -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  cf.br ^bb2(%[[B1]] : none)
//       CHECK:  ^bb2(%[[B2:.*]]: none):
//       CHECK:  py_ir.return %[[B2]] : none
py_ir.module {
  %f = py_ir.func "foo" -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    cf.br ^bb2
   ^bb2:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  cf.cond_br %{{.*}}, ^bb1(%[[B1]] : none), ^bb2(%[[B1]] : none)
//       CHECK:  ^bb2(%[[B2:.*]]: none):
//       CHECK:  py_ir.return %[[B2]] : none
py_ir.module {
  %f = py_ir.func "foo" -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    %n = py_ir.none
    %c = py_ir.cast %n : none to i1
    cf.cond_br %c, ^bb1, ^bb2
   ^bb2:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  cf.cond_br %{{.*}}, ^bb2(%[[B1]] : none), ^bb2(%[[B1]] : none)
//       CHECK:  ^bb2(%[[B2:.*]]: none):
//       CHECK:  py_ir.return %[[B2]] : none
py_ir.module {
  %f = py_ir.func "foo" -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    %n = py_ir.none
    %c = py_ir.cast %n : none to i1
    cf.cond_br %c, ^bb2, ^bb2
   ^bb2:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}
