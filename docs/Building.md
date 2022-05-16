# Building the code

## Prerequisites

```
cmake 3.17.5
ninja 1.8.2
Xilinx Vitis 2020.1
sudo pip3 install joblib psutil
clang/llvm 14+ from source https://github.com/llvm/llvm-project
Xilinx cmakeModules from https://github.com/Xilinx/cmakeModules
```

In addition, the following optional packages may be useful
```
LibXAIE is a backend target used to execute designs in hardware: https://github.com/Xilinx/embeddedsw/tree/master/XilinxProcessorIPLib/drivers/aiengine
```

Currently, the only supported target is the Xilinx VCK190 board, running Ubuntu-based Linux, however
the tools are largely board and device independent and can be adapted to other environments.

## Building on X86

First compile LLVM, with the ability to target AArch64 as a cross-compiler, and with MLIR enabled:
In addition, we make some common build optimizations to use a linker ('lld' or 'gold') other than
'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang
so.  You may find that other options are also useful.  Note that due to changing MLIR APIs, only a
particular revision is expected to work.  See utils/clone-llvm.sh for the correct commithash.

```sh
git clone https://github.com/llvm/llvm-project
cd llvm-project
git checkout ${commithash}
mkdir ${LLVMBUILD}; cd ${LLVMBUILD}
cmake -GNinja \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DCLANG_LINK_CLANG_DYLIB=ON \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=${ACDCInstallDir} \
    -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
    -DLLVM_TARGETS_TO_BUILD:STRING="X86;ARM;AArch64;" \
    ../llvm
ninja; ninja check-llvm; ninja install
```

Then you can build the AIE tools:
```sh
git clone https://github.com/Xilinx/cmakeModules
git clone https://github.com/Xilinx/mlir-aie
mkdir build; cd build
cmake -GNinja \
    -DLLVM_DIR=${absolute path to LLVMBUILD}/lib/cmake/llvm \
    -DMLIR_DIR=${absolute path to LLVMBUILD}/lib/cmake/mlir \
    -DLibXAIE_DIR=${absolute path to LibXAIE} \
    -DCMAKE_MODULE_PATH=${absolute path to cmakeModules}/ \
    -DVitisSysroot=${SYSROOT} \
    -DCMAKE_BUILD_TYPE=Debug \
    ..
ninja; ninja check-aie; ninja mlir-doc; ninja install
```

The MLIR AIE tools will be able to generate binaries targetting a combination of AIEngine and ARM processors.

### Sysroot
Since the AIE tools are cross-compiling, in order to actually compile code, we need a 'sysroot' directory,
containing an ARM rootfs.  This rootfs must match what will be available in the runtime environment.
Note that copying the rootfs is often insufficient, since many root file systems include absolute links.
Absolute symbolic links can be converted to relative symbolic links using [symlinks](https://github.com/brandt/symlinks).

```sh
cd /
sudo symlinks -rc .
```
Following the [platform build steps](Platform.md) will also create a sysroot.

## Environment setup
In order to run all the tools, it may be necessary to add some paths into your environment:

```
setenv MLIRAIE /path/to/build/install
setenv PATH ${MLIRAIE}/bin:${PATH}
setenv PYTHONPATH ${MLIRAIE}/python:${PYTHONPATH}
setenv LD_LIBRARY_PATH ${MLIRAIE}/lib:${LD_LIBRARY_PATH}
```

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>
