// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CompilerFront.hpp"

namespace py = pybind11;

PYBIND11_MODULE(compiler, m) {
  m.def("compile_ast", &compileAST, "compile_ast", py::arg("source"),
        py::arg("func_name"));
}
