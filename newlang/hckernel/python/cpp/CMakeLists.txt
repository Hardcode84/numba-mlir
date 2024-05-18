# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(pybind11 REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(python_module_name compiler)

pybind11_add_module(${python_module_name} MODULE
    src/CompilerFront.cpp
    src/CompilerFront.hpp
    src/PyModule.cpp
)

target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(${python_module_name} PRIVATE
    hc_core
    hc_frontend
    MLIRPass
    )

apply_llvm_compile_flags(${python_module_name})

target_include_directories(${python_module_name} PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    )

install(TARGETS ${python_module_name} DESTINATION ${CMAKE_INSTALL_PREFIX})
