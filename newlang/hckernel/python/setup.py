import os
import subprocess
from setuptools import setup, find_packages

root_dir = os.path.dirname(os.path.abspath(__file__))
cmake_build_dir = os.path.join(root_dir, "build")
cmake_dir = os.path.join(root_dir, "..")
install_dir = os.path.join(root_dir, "hckernel", "_native")

env = os.environ


def get_env(env_name):
    env_val = env.get(env_name)

    if env_val is None:
        raise RuntimeError(f"Enviroment variable '{env_name}' is not set")

    return env_val


LLVM_PATH = get_env("LLVM_PATH")
LLVM_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "llvm")
MLIR_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "mlir")
LIT_DIR = os.path.join(root_dir, "..", "scripts", "runlit.py")

cmake_cmd = [
    "cmake",
    cmake_dir,
    "-GNinja",
    "-DCMAKE_INSTALL_PREFIX=" + install_dir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLVM_DIR=" + LLVM_DIR,
    "-DMLIR_DIR=" + MLIR_DIR,
    "-DHC_ENABLE_TESTS=ON",
    "-DLLVM_EXTERNAL_LIT=" + LIT_DIR,
]

subprocess.run(["mkdir", cmake_build_dir])
subprocess.run(["mkdir", install_dir])

subprocess.check_call(
    cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir, env=env
)
subprocess.check_call(
    ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir, env=env
)
subprocess.check_call(
    ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir, env=env
)

setup(
    name="hckernel",
    version="1.0",
    packages=find_packages(where=root_dir, include=["*", "hckernel", "hckernel.*"]),
)
