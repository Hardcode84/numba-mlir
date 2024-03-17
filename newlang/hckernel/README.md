## Building

```
mkdir cmake_build && cd cmake_build
cmake -G Ninja -Dpybind11_DIR:PATH=<pybin11-install>/share/cmake/pybind11 -DLLVM_DIR:PATH=<llvm-install>/lib/cmake/llvm -DMLIR_DIR:PATH=<llvm-install>/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release -DHCKERNEL_ENABLE_TESTS=ON -DLLVM_EXTERNAL_LIT=<path-to-repo>\hckernel\scripts\runlit.py ..
ninja check check-hckernel
```
