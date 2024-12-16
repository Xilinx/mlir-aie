# Building the code

## Prerequisites

```
lld
cmake 3.20.6
ninja 1.8.2
Xilinx Vitis 2023.2
python 3.8.x and pip
virtualenv
pip3 install psutil rich pybind11 numpy
clang/llvm 14+ from source https://github.com/llvm/llvm-project
```

Xilinx Vitis can be downloaded and installed from the [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) site.

In order to successfully install Vitis on a fresh bare-bones Ubuntu install, some additional prerequisites are required, [documented here](https://support.xilinx.com/s/article/63794?language=en_US). For Ubuntu 20.04, the installation should succeed if you additionally install the following packages: `libncurses5 libtinfo5 libncurses5-dev libncursesw5-dev ncurses-compat-libs libstdc++6:i386 libgtk2.0-0:i386 dpkg-dev:i386 python3-pip` Further note that the above mentioned cmake prerequisite is _not_ satisfied by the package provided by Ubuntu; you will need to obtain a more current version.

NOTE: Using the Vitis recommended `settings64.sh` script to set up your environement can cause tool conflicts. Setup your environment in the following order for aietools and Vitis:
 
```
export PATH=$PATH:<Vitis_install_path>/Vitis/2023.2/aietools/bin:<Vitis_install_path>/Vitis/2023.2/bin
```

The cmake and python packages prerequisites can be satisfied by sourcing the `utils/setup_python_packages.sh` script. See step 2 of the build instructions. 
This script requires `virtualenv`.

clang/llvm 14+ are recommended to be built with the provided scripts. See step 3. of the build instructions. 

When targetting the VCK5000 Versal device, you must build and install our experimental ROCm runtime which allows us to communicate with the AIEs. The [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms) repository contains documentation on how to install our experimental ROCm runtime. When targetting the VCK5000, it will be necessary to [install a global version of ROCM 5.6](https://rocm.docs.amd.com/en/docs-5.6.0/deploy/linux/os-native/install.html). Details of all these steps can be found in the [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms#getting-started) repo. 

## Building on X86 for mlir-aie development

1. Clone the `mlir-aie` repository with its sub-modules:
    ```
    git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
    cd mlir-aie
    ```

    __All subsequent steps should be run from inside the top-level
    directory of the `mlir-aie` repository cloned above.__

2. Source `utils/setup_python_packages.sh` to setup the prerequisite python
    packages. This script creates and installs the python packages
    listed in `utils/requirements.txt` in a virtual python environment
    called 'sandbox', then it enters the sandbox:
    ```
    source utils/setup_python_packages.sh
    ```

    If you need to exit the sandbox later, type `deactivate`.  If you
    have a recent Linux distribution, you might not need this, as you
    are able to have all the required packages from the distribution.

3. Clone and compile LLVM, with the ability to target AArch64 as a
   cross-compiler, and with MLIR enabled: in addition, we make some
   common build optimizations to use a linker (`lld` or `gold`) other
   than `ld` (which tends to be quite slow on large link jobs) and to
   link against `libLLVM.so` and `libClang.so`. You may find that other
   options are also useful. Note that due to changing MLIR APIs, only
   a particular revision is expected to work.

    To clone `llvm`, run `utils/clone-llvm.sh` (see
    `utils/clone-llvm.sh` for the correct `llvm` commit hash):
    ```
    ./utils/clone-llvm.sh
    ```

    If you have already an LLVM repository, you can instead of cloning
    just make a new worktree from it by using:
    ```
    ./utils/clone-llvm.sh --llvm-worktree <directory-of-existing-LLVM-repository>
    ```

    To build (compile and install) LLVM, run `utils/build-llvm-local.sh` in the directory where `llvm` has
    been cloned. See `utils/build-llvm-local.sh` for additional shell script arguments.
    (Note that `build-llvm-local.sh` and `build-llvm.sh` are a
    variation of the LLVM build script used for CI on GitHub and
    looking at the continuous integration recipe
    https://github.com/Xilinx/mlir-aie/blob/main/.github/workflows/buildAndTest.yml
    and output https://github.com/Xilinx/mlir-aie/actions/ might help
    in the case of compilation problem.)
    ```
    ./utils/build-llvm-local.sh
    ```
    This will build LLVM in `llvm/build` and install the LLVM binaries under `llvm/install`.

4. Build the MLIR-AIE tools by calling `utils/build-mlir-aie.sh` or `utils/build-mlir-aie-pcie.sh` 
    for Ryzen AI or Versal respectively  with the path to the `llvm/build` directory. 
    The Vitis environment will have to be set up for this to succeed.  

    Building mlir-aie for a Ryzen AI system:

    ```
    source <Vitis Install Path>/settings64.sh
    ./utils/build-mlir-aie.sh <llvm dir>/<build dir>
    ```

    Building mlir-aie for a VCK5000 system:

    ```
    source <Vitis Install Path>/settings64.sh
    ./utils/build-mlir-aie-pcie.sh <llvm dir>/<build dir>
    ```

    This will create a `build` and `install` folder in the directory that you cloned MLIR AIE into. 

    The MLIR AIE tools will be able to generate binaries targetting a combination of AIEngine and ARM/x86 processors. 

5. In order to run all the tools, it is necessary to add some paths into your environment. This can be
done by sourcing the `utils/env_setup.sh` script with the paths to the install folders for mlir-aie
and llvm.
    ```
    source utils/env_setup.sh <mlir-aie>/install <llvm dir>/install
    ```

6. **Only for the VCK5000:** The PCIe AIR runtime requires the use of the [AIR PCIe kernel driver](https://github.com/Xilinx/ROCm-air-platforms/tree/main/driver). This directory contains documentation on how to compile and load the AIR PCIe kernel driver.

Note that when coming back to this install with a fresh environment, it is necessary to rerun the `utils/env_setup.sh` script to setup your environment as well as activate the Python virtual environment using the following command.
```
source ironenv/bin/activate
```

## Building on X86 targetting the VCK5000 with seperate builds of ROCm runtime and aie-rt

The `build-mlir-aie-pcie.sh` script will automatically build a local install of the ROCm Runtime and aie-rt if the corresponding install directories are not provided. Another option is to build these externally and then point `build-mlir-aie-pcie.sh` to where they are installed. Below are the steps for how to build the tools using this second option.

We chose to install the aie-rt library in /opt/xaiengine but it is not required for the tools to be installed there. Just ensure that when building mlir-aie and mlir-air, that you point to the directory in which the aie-rt library was installed. Below are the steps to build the aie-rt library and move the installation to /opt/xaiengine.

```
git clone https://github.com/stephenneuendorffer/aie-rt
cd aie-rt
git checkout phoenix_v2023.2
cd driver/src
make -f Makefile.Linux CFLAGS="-D__AIEAMDAIR__"
sudo cp -r ../include /opt/xaiengine/
sudo cp libxaiengine.so* /opt/xaiengine/lib/
export LD_LIBRARY_PATH=/opt/xaiengine/lib:${LD_LIBRARY_PATH}
```

To install our experimental ROCm runtime  which allows us to communicate with the AIEs on the VCK5000, it is necessary to [install a global version of ROCM 5.6](https://rocm.docs.amd.com/en/docs-5.6.0/deploy/linux/os-native/install.html). Details of all these steps can be found in the [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms#getting-started). Afterwards, follow the steps in [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms#getting-started) to build the experimental ROCm runtime that targets the AIEs in the VCK5000. You can run the following script to clone the ROCm-air-platforms repository:

```
./utils/clone-rocm-air-platforms.sh
```

Then, set `${ROCM_ROOT}` to the ROCm install. Then, run the following command to build the mlir-aie toolchain targetting the VCK5000 pointing to the externally installed aie-rt and experimental ROCm runtime.

```
./utils/build-mlir-aie-pcie.sh llvm/build/ build install /opt/xaiengine ${ROCM_ROOT}/lib/cmake/hsa-runtime64/ ${ROCM_ROOT}/lib/cmake/hsakmt/
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
Following the [platform build steps](Platform.md) will create such a sysroot based on PetaLinux. Note that those instructions require Vitis 2021.2 -- building a sysroot with Vitis 2023.2 will not currently succeed.

-----

<p align="center">Copyright&copy; 2019-2023 AMD/Xilinx</p>
