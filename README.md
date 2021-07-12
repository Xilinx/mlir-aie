This repository contains an MLIR-based toolchain for Xilinx Versal AIEngine-based devices.  This can be used to generate low-level configuration for the AIEngine portion of the device, including processors, stream switches, TileDMA and ShimDMA blocks. Backend code generation is included, targetting the LibXAIE library.  This project is primarily intended to support tool builders with convenient low-level access to devices and enable the development of a wide variety of programming models from higher level abstractions.  As such, although it contains some examples, this project is not intended to represent end-to-end compilation flows or to be particularly easy to use for system design.

# Building ACDC

## Prerequisites

```
cmake 3.17.5
ninja 1.8.2
Xilinx Vitis 2020.2
sudo pip3 install joblib psutil
clang/llvm 13+ from source https://github.com/llvm/llvm-project/commit/ebe408ad8003c946ef871b955ab18e64e82697cb
```

Currently, the only supported target is the Xilinx VCK190 board, running Ubuntu-based Linux.

## Building on X86

First compile LLVM, with the ability to target AArch64 as a cross-compiler, and with MLIR enabled:
In addition, we make some common build optimizations to use a linker other than 'ld' (which tends
to be quite slow on large link jobs) and to link against libLLVM.so and libClang.so.  You may find
that other options are also useful.
```sh
mkdir ${LLVM}; cd ${LLVM}
cmake -GNinja \
    -DLLVM_LINK_LLVM_DYLIB=ON 
    -DCLANG_LINK_CLANG_DYLIB=ON
    -DLLVM_BUILD_UTILS=ON
    -DLLVM_INSTALL_UTILS=ON
    -DLLVM_USE_LINKER=lld  (or gold)
    -DCMAKE_INSTALL_PREFIX=${ACDCInstallDir}
    -DLLVM_ENABLE_PROJECTS="clang;lld;mlir"
    -DLLVM_TARGETS_TO_BUILD:STRING="X86;ARM;AArch64;"
    ..
ninja; ninja check-llvm; ninja install
```

Then you can build the AIE tools:
```sh
mkdir build; cd build
cmake -GNinja \
    -DLLVM_DIR=${LLVM}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM}/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=/wrk/hdstaff/stephenn/acdc/cmakeModules/cmakeModulesXilinx/ \
    -DVitisSysroot=${SYSROOT} \
    -DCMAKE_BUILD_TYPE=Debug \
    ..
ninja; ninja check-aie; ninja mlir-doc; ninja install
```

The AIE tools will be able to generate binaries targetting a combination of AIEngine and ARM processors.

### Sysroot
Since the AIE tools are cross-compiling, in order to actually compile code, we need a 'sysroot' directory,
containing an ARM rootfs.  This rootfs must match what will be available in the runtime environment.
Note that copying the rootfs is often insufficient, since many root file systems include absolute links.
Absolute symbolic links can be converted to relative symbolic links using [symlinks](https://github.com/brandt/symlinks).

```sh
cd /
sudo symlinks -rc .
```


## Environment setup
In order to run all the tools, it may be necessary to add some paths into your environment:

```
setenv ACDC /path/to/build/install
setenv PATH ${ACDC}/bin:${PATH}
setenv PYTHONPATH ${ACDC}/python:${PYTHONPATH}
setenv LD_LIBRARY_PATH ${ACDC}/lib:${LD_LIBRARY_PATH}
```

