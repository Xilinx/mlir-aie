# Building the code

## Prerequisites

```
clang 10.0.0+
lld
cmake 3.20.6
ninja 1.8.2
Xilinx Vitis 2022.2
python 3.8.x and pip
pip3 install psutil rich pybind11 numpy
clang/llvm 14+ from source https://github.com/llvm/llvm-project
Xilinx cmakeModules from https://github.com/Xilinx/cmakeModules
```

Xilinx Vitis can be downloaded and installed from the [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) site. 
NOTE: using the Vitis recommended settings64.sh script to set up your environement can cause tool conflicts. Setup your environment in the following order for aietools and Vitis:
 
```
export PATH=$PATH:<Vitis_install_path>/Vitis/2022.2/aietools/bin:<Vitis_install_path>/Vitis/2022.2/bin
```

The cmake and python packages prerequisites can be satisfied by sourcing the setup_python_packages.sh script. See step 2. of the build instructions. 
This script requires `virtualenv`.

clang/llvm 14+ are recommended to be built with the provided scripts. See step 3. of the build instructions. 

In addition, the following optional packages may be useful:
```
LibXAIE is a backend target used to execute designs in hardware: https://github.com/Xilinx/embeddedsw/tree/master/XilinxProcessorIPLib/drivers/aiengine
```
Note that if you build one of the supported platforms like vck190_bare_prod, the generated sysroot 
already contains the LibXAIE drivers so you do not need to download the embeddedsw repo or 
define the LibXAIE_DIR cmake parameter.

Currently, the only supported target is the Xilinx VCK190 board, running Ubuntu-based Linux, however
the tools are largely board and device independent and can be adapted to other environments.

## Building on X86

1. Clone the mlir-aie repo.
    ```
    git clone https://github.com/Xilinx/mlir-aie.git
    cd mlir-aie
    ```

    __All subsequent steps should be run from inside the top-level directory of the mlir-aie repo cloned above.__

2. Run utils/setup_python_packages.sh to setup the prerequisite python packages. This script creates and installs the python packages listed in utils/requirements.txt in a virtual python environment called 'sandbox'.
    ```
    source utils/setup_python_packages.sh
    ```

3. Clone and compile LLVM, with the ability to target AArch64 as a cross-compiler, and with MLIR 
enabled: in addition, we make some common build optimizations to use a linker ('lld' or 'gold') other 
than 'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang
so. You may find that other options are also useful. Note that due to changing MLIR APIs, only a
particular revision is expected to work.  

    To clone llvm and cmakeModules, run utils/clone-llvm.sh (see utils/clone-llvm.sh for the correct llvm commithash).
    ```
    ./utils/clone-llvm.sh
    ```
    To build (compile and install) llvm, run utils/build-llvm-local.sh in the directory that llvm and 
    cmakeModules are cloned in. See build-llvm-local.sh for additional shell script arguments. 
    Note that build-llvm.sh is a variation of the llvm build script used for CI on github.
    ```
    ./utils/build-llvm-local.sh 
    ```
    This will build llvm in llvm/build and install the llvm binaries under llvm/install.

4. Build the mlir-aie tools by calling utils/build-mlir-aie.sh with paths to the 
llvm/build and cmakeModules repos (note that clone-llvm.sh puts the cmakeModules repo under 
cmakeModules/cmakeModulesXilinx). 
    ```
    ./utils/build-mlir-aie.sh <llvm dir>/<build dir> <cmakeModules dir>/cmakeModulesXilinx
    ```
    This will create a build and install folder under /mlir-aie. 

    The MLIR AIE tools will be able to generate binaries targetting a combination of AIEngine and ARM processors.

5. In order to run all the tools, it is necessary to add some paths into your environment. This can be 
done by calling the utils/env_setup.sh script with the paths to the install folders for mlir-aie
and llvm.
    ```
    source utils/env_setup.sh <mlir-aie>/install <llvm dir>/install
    ```

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

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>
