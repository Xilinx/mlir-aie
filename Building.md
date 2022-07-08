# Building the code

## Prerequisites

```
cmake 3.17.5
ninja 1.8.2
Xilinx Vitis 2021.2
sudo pip3 install joblib psutil
clang/llvm 14+ from source https://github.com/llvm/llvm-project
Xilinx cmakeModules from https://github.com/Xilinx/cmakeModules
```

In addition, the following optional packages may be useful
```
LibXAIE is a backend target used to execute designs in hardware: https://github.com/Xilinx/embeddedsw/tree/master/XilinxProcessorIPLib/drivers/aiengine
```
Note that if you build one of the supported platforms like vck190_bare_prod, the generated sysroot 
already contains the LibXAIE drivers so you do not need to download the embeddedsw repo or 
define the LibXAIE_DIR cmake parameter.

Currently, the only supported target is the Xilinx VCK190 board, running Ubuntu-based Linux, however
the tools are largely board and device independent and can be adapted to other environments.

## Building on X86

This mlir-aie repo should already cloned locally (<mlir-aie> : absolute directory to mlir-aie).

First clone and compile LLVM, with the ability to target AArch64 as a cross-compiler, and with MLIR 
enabled: In addition, we make some common build optimizations to use a linker ('lld' or 'gold') other 
than 'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang
so.  You may find that other options are also useful.  Note that due to changing MLIR APIs, only a
particular revision is expected to work.  

To clone llvm and cmakeModules, see utils/clone-llvm.sh for the correct commithash.
```
clone-llvm.sh
```
To build (compile and install) llvm, run utils/build-llvm-local.sh in the directory that llvm and 
cmakeModules are cloned in. See build-llvm-local.sh for additional shell script arguments. 
Note that build-llvm.sh is a variation of the llvm build script used for CI on github.
```
build-llvm-local.sh 
```
This will build llvm in llvm/build and install the llvm binaries under llvm/install

Finally, build the mlir-aie tools by calling utils/build-mlir-aie.sh with absolute paths to the 
sysroot, llvm and cmakeMdoules repos (note that clone-llvm.sh puts the cmakeModules repo under cmakeModules/cmakeModulesXilinx). 
```
build-mlir-aie.sh <sysroot dir> <llvm dir> <cmakeModules dir>/cmakeModulesXilinx
```
This will create a build and install folder under <mlir-aie>. 

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
In order to run all the tools, it may be necessary to add some paths into your environment. This can be 
done by calling the utils/env_setup.sh script with the absolute path to the install folder for mlir-aie
and llvm.
```
source <mlir-aie>/utils/env_setup.sh <mlir-aie>/install <llvm dir>/install
```

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>
