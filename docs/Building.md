# Building the MLIR-AIE Codebase on Linux

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 24.04** or **Ubuntu 24.10** install. It is possible to use **Ubuntu 22.04** however you must follow the documentation on the [xdna-driver](https://github.com/amd/xdna-driver) repository to configure the Linux kernel, driver and runtime for deployment. 

## Initial Setup

  > Be sure you have the latest BIOS on your laptop or mini-PC that enables the NPU. See [here](#update-bios).

If starting from `Ubuntu 24.04` you may need to update the Linux kernel to 6.11+ by installing the Hardware Enablement (HWE) stack:

  ```bash
  sudo apt update 
  sudo apt install --install-recommends linux-generic-hwe-24.04
  sudo reboot
  ```

## Prerequisites

### BIOS Settings:

Turn off SecureBoot (Allows for unsigned drivers to be installed):
   ```BIOS → Security → Secure boot → Disable```

### Build and install the XDNA™ Driver and XRT

1. Execute the scripted build process:

    > This script will install package dependencies, build the xdna-driver and xrt packages, and install them. *These steps require `sudo` access.*

    ```bash
    bash ./utils/build_drivers.sh
    ```

1. Reboot as directed after the script exits. 

    ```bash
    sudo reboot
    ```

1. Check that the NPU is working if the device appears with xrt-smi:

   ```bash
   source /opt/xilinx/xrt/setup.sh
   xrt-smi examine
   ```

   > At the bottom of the output you should see:
   >  ```
   >  Devices present
   >  BDF             :  Name             
   > ------------------------------------
   >  [0000:66:00.1]  :  NPU Strix
   >  ```

### Install AIETools

1. Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Access](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab). We will assume you use the installation directory, `/tools/ryzen_ai-1.3.0/vitis_aie_essentials`.

   > This is an early access lounge, you must register and be granted access at this time.

    1. Download VAIML Installer for Linux based compilation: `ryzen_ai-1.3.0ea1.tgz`

    1. Extract the required tools:

       ``` bash
          tar -xzvf ryzen_ai-1.3.0ea1.tgz
          cd ryzen_ai-1.3.0
          mkdir vitis_aie_essentials
          mv vitis_aie_essentials*.whl vitis_aie_essentials
          cd vitis_aie_essentials
          unzip vitis_aie_essentials*.whl
       ```

1. Set up an AI Engine license.

    1. Get a local license for AI Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).

    1. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`:


1. Setup your environment using the following script for Vitis™ for AIETools:

   ```bash
   #!/bin/bash
   #################################################################################
   # Setup Vitis AIE Essentials
   #################################################################################
   export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
   export PATH=$PATH:${AIETOOLS_ROOT}/bin
   export LM_LICENSE_FILE=/opt/Xilinx.lic
   ```

### Install IRON and MLIR-AIE Prerequisites

1. Install the following packages needed for MLIR-AIE:

    ```bash
    # Python versions 3.10, 3.12 and 3.13 are currently supported by our wheels
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip
    ```

## Build and Install mlir-aie and IRON

1. Clone [the mlir-aie repository](https://github.com/Xilinx/mlir-aie.git):
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   git submodule update --init --recursive
   ```

1. Setup a virtual environment:
   ```bash
   python3 -m venv ironenv
   source ironenv/bin/activate
   python3 -m pip install --upgrade pip
   ```

1. Install required Python packages:
   ```bash
   # Install basic Python requirements 
   python3 -m pip install -r python/requirements.txt
   ```

1. Install Python packages required for development and testing:
   ```bash
   # Install Python requirements for development and testing
   python3 -m pip install -r python/requirements_dev.txt

   # This installs the pre-commit hooks defined in .pre-commit-config.yaml
   pre-commit install
   ```

1. Use scripted mlir-aie build process. This script downloads llvm/mlir from wheels before building.
   ```bash
   bash ./utils/build-mlir-aie-from-wheels.sh
   ```

1. Setup environment
   ```bash
   source utils/env_setup.sh install
   ```

## Build an IRON Design for AIEs in the AMD Ryzen™ AI NPU

For your design of interest, for instance from [programming_examples](../programming_examples/), 2 steps are needed: (i) build the AIE design and then (ii) build the host code.

### Build Device AIE Part

1. Goto the design of interest and run:
   ```bash
   make
   ```

1. Build host code and execute the design:
    ```bash
    make run
    ```

## Building without Vitis (Peano-only)

For users who prefer to use only open-source tools, MLIR-AIE can be built using only the [Peano](https://github.com/Xilinx/llvm-aie) compiler (llvm-aie) instead of Vitis AIE Essentials. This enables:

- **Open-source compilation**: No proprietary tools or licenses required
- **AIE2 and AIE2P support**: Supports Ryzen AI NPUs (Phoenix, Strix, Strix Halo, Krackan)

> **Note**: Peano-only builds do NOT support the original Versal AIE architecture. Only AIE2 (NPU1) and AIE2P (NPU2) targets are available.

### Peano-only Build Steps

1. Install Peano (llvm-aie) by downloading the wheel from [llvm-aie releases](https://github.com/Xilinx/llvm-aie/releases):
   ```bash
   pip download llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/latest-peano
   unzip llvm_aie*.whl -d peano_install
   export PEANO_INSTALL_DIR=$PWD/peano_install/llvm_aie
   ```

2. Configure with the Peano-only build option:
   ```bash
   mkdir build && cd build
   cmake -GNinja \
     -DAIE_PEANO_ONLY_BUILD=ON \
     -DPEANO_INSTALL_DIR=$PEANO_INSTALL_DIR \
     -DCMAKE_INSTALL_PREFIX=../install \
     -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
     -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm \
     ..
   ninja install
   ```

3. When compiling AIE kernels, use the `--no-xchesscc` flag:
   ```bash
   cd programming_examples/basic/passthrough_kernel
   make AIECC_FLAGS="--no-xchesscc --no-xbridge"
   ```

### CMake Options for Peano-only Builds

| Option | Description |
|--------|-------------|
| `AIE_PEANO_ONLY_BUILD` | Enable Peano-only build (default: OFF) |
| `PEANO_INSTALL_DIR` | Path to Peano (llvm-aie) installation |

When `AIE_PEANO_ONLY_BUILD=ON`:
- `AIE_COMPILER` is automatically set to `PEANO`
- `AIE_LINKER` is automatically set to `PEANO`
- Only AIE2 and AIE2P subdirectories in `aie_runtime_lib/` are built
- The `chess_intrinsic_wrapper.ll` is not generated (not needed for Peano)

## Learn more

1. Additional MLIR-AIE documentation is available on the [website](https://xilinx.github.io/mlir-aie/)

1. AIE API header library documentation for single-core AIE programming in C++ is avaiable [here](https://xilinx.github.io/aie_api/topics.html)

## Contributing:

Interested in contributing MLIR-AIE? [Information for developers](./CONTRIBUTING.md)

## Troubleshooting:

### Update BIOS:

Be sure you have the latest BIOS for your laptop or mini PC, this will ensure the NPU (sometimes referred to as IPU) is enabled in the system. You may need to manually enable the NPU:
   ```Advanced → CPU Configuration → IPU``` 

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu.

[MLIR Dialect and Compiler Documentation](https://xilinx.github.io/mlir-aie/)

-----

<p align="center">Copyright&copy; 2019-2025 Advanced Micro Devices, Inc</p>
