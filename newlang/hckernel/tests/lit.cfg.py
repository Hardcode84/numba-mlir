# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "hckernel-mlir-tests"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".pymlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.hckernel_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%shlibprefix", config.hckernel_shlib_prefix))
config.substitutions.append(("%mlir_wrappers_dir", config.mlir_wrappers_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP", "CONDA_PREFIX"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.hckernel_obj_root, "tests")
config.pyfront_dir = os.path.join(
    config.hckernel_obj_root, "tools", "pyfront"
)

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.llvm_tools_dir, config.pyfront_dir]
tools = ["pyfront"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
