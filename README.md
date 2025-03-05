# MLIR-based AI Engine toolchain

[![Build and Test across Python versions](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestPythons.yml/badge.svg)](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestPythons.yml)

[![Build and Test with AIE tools on Ryzen AI](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestRyzenAI.yml/badge.svg)](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestRyzenAI.yml)

[![Compile across platforms](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestMulti.yml/badge.svg)](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestMulti.yml)

![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/mlir-aie)

![](https://mlir.llvm.org//mlir-logo.png)

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for AI Engine-enabled devices, such as [AMD Ryzen™ AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) and [Versal™](https://www.xilinx.com/products/technology/ai-engine.html).  This repository can be used to generate low-level configurations for the AI Engine portion of these devices. AI Engines are organized as a spatial array of tiles, where each tile contains AI Engine cores and/or memories. The spatial array is connected by stream switches that can be configured to route data between AI Engine tiles scheduled by their programmable Data Movement Accelerators (DMAs). This repository contains MLIR representations, with multiple levels of abstraction, to target AI Engine devices. This enables compilers and developers to program AI Engine cores, as well as describe data movements and array connectivity. A Python API is made available as a convenient interface for generating MLIR design descriptions. Backend code generation is also included, targeting the [aie-rt](https://github.com/Xilinx/aie-rt/tree/main-aie) library.

This project is primarily intended to support the open-source community, particularly tool builders, with low-level access to AIE devices and enable the development of a wide variety of programming models from higher level abstractions. We provide an example programming flow: Interface Representation for hands-ON (IRON) close-to-metal programming of the AIE-array. IRON is an open access toolkit enabling performance engineers to build fast and efficient, often specialized designs through a set of Python language bindings around the mlir-aie dialect. As such, it contains some examples, however this project is not intended to represent an end-to-end compilation flow for all application designs. If you're looking for an out-of-the-box experience for highly efficient machine learning, check out the [AMD Ryzen™ AI Software Platform](https://github.com/amd/RyzenAI-SW/).

# Getting Started for AMD Ryzen™ AI on Linux

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 24.04** or **Ubuntu 24.10** install.

## Initial Setup

  > Be sure you have the latest BIOS on your laptop or mini-PC that enables the NPU. See [here](#update-bios).

If starting from `Ubuntu 24.04` you may need to update the Linux kernel by installing the Hardware Enablement (HWE) stack:

  ```bash
  sudo apt update 
  sudo apt install --install-recommends linux-generic-hwe-24.04
  sudo reboot
  ```

## Prerequisites

### BIOS Settings:

Turn off SecureBoot (Allows for unsigned drivers to be installed):
   ```BIOS → Security → Secure boot → Disable```

### Install the XDNA™ Driver

1. Clone the XDNA™ driver repository and its submodules.
    ```bash
    git clone https://github.com/amd/xdna-driver.git
    export XDNA_SRC_DIR=$(realpath xdna-driver)
    cd xdna-driver
    git submodule update --init --recursive
    ```

    > The submodules use SSH remotes. You will need a GitHub account and locally installed SSH keys to pull the submodules. Follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to set up an SSH key. Alternatively, edit `.gitmodules` to use HTTPS instead of SSH.

1. Install XRT. (Below steps are adapted from [here](https://xilinx.github.io/XRT/master/html/build.html).)

    1. Install XRT prerequisites.
    
       ```bash
       cd $XDNA_SRC_DIR
       sudo ./tools/amdxdna_deps.sh
       ```

    2. Build XRT.

       ```bash
       cd $XDNA_SRC_DIR/xrt/build
       ./build.sh -npu -opt
       ```

    3. Install XRT.

       ```bash
       cd $XDNA_SRC_DIR/xrt/build/Release
       # Ubuntu 24.04
       sudo apt reinstall ./xrt_202510.2.19.0_24.04-amd64-base.deb
       sudo apt reinstall ./xrt_202510.2.19.0_24.04-amd64-base-dev.deb
       # Ubuntu 24.10
       sudo apt reinstall ./xrt_202510.2.19.0_24.10-amd64-base.deb
       sudo apt reinstall ./xrt_202510.2.19.0_24.10-amd64-base-dev.deb
       ```

       > **An error might occur during this proces.** If so, you may have to remove and force-overwrite/reinstall the packages.

1. Build XDNA-Driver. Below steps are adapted from [here](https://github.com/amd/xdna-driver).

    ```bash
    cd $XDNA_SRC_DIR/build
    ./build.sh -release
    ./build.sh -package
    ```

1. Install XDNA™.

    ```bash
    cd $XDNA_SRC_DIR/build/Release
    # Ubuntu 24.04
    sudo apt reinstall ./xrt_plugin.2.19.0_ubuntu24.04-x86_64-amdxdna.deb
    # Ubuntu 24.10
    sudo apt reinstall ./xrt_plugin.2.19.0_ubuntu24.10-x86_64-amdxdna.deb
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
   >  [0000:66:00.1]  :  RyzenAI-npu1
   >  ```

### Install IRON and MLIR-AIE Prerequisites

1. Install the following packages needed for MLIR-AIE:

    ```bash
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake python3-venv python3-pip libxrender1 libxtst6 libxi6
    ```

1. Install g++13 and opencv which is needed for some programming examples:

   ```bash
   sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   sudo apt update
   sudo apt install gcc-13 g++-13 -y
   sudo apt install libopencv-dev python3-opencv
   ```

## Install IRON for AMD Ryzen™ AI AIE Application Development

1. Clone [the mlir-aie repository](https://github.com/Xilinx/mlir-aie.git):
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ```

1. Source `utils/quick_setup.sh` to setup the prerequisites and
   install the mlir-aie compiler tools from whls.

## Build an IRON Design for AIEs in the AMD Ryzen™ AI NPU

> Remember to set up your environment including IRON, and XRT
> ```
>   source /opt/xilinx/xrt/setup.sh
>   source ironenv/bin/activate
>   source utils/env_setup.sh  
> ```
> 
For your design of interest, for instance from [programming_examples](../programming_examples/), 2 steps are needed: (i) build the AIE design and then (ii) build the host code.

### Build Device AIE Part

1. Goto the design of interest and run `make`

### Build and Run Host Part

1. Build: Goto the same design of interest folder where the AIE design just was built (see above)
    ```bash
    make <testName>.exe
    ```
    > Note that the host code target has a `.exe` file extension even on Linux. Although unusual, this is an easy way for us to distinguish whether we want to compile device code or host code.


1. Run (program arguments are just an example for vector_scalar_add design)
    ```bash
    make run
    ```

## Learn more about NPU programming with IRON

1. Continue to the [IRON AIE Application Programming Guide](programming_guide)

1. Some MLIR-AIE documentation is available on the [website](https://xilinx.github.io/mlir-aie/)

## Optional: Install AIETools

> You may skip the Vitis™ installation step if you intend to only target AMD XDNA™/AIE-ML (AIE2) and AMD XDNA™ 2 (AIE2P) using our open-source single-core compiler [Peano](https://github.com/Xilinx/llvm-aie). Compiling with `xchesscc` is not supported without installing AMD Vitis™ AIE Essentials. 

1. Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Accesss](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab). We will assume you use the installation directory, `/tools/ryzen_ai-1.3.0/vitis_aie_essentials`.

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

## Troubleshooting:

### Update BIOS:

Be sure you have the latest BIOS for your laptop or mini PC, this will ensure the NPU (sometimes referred to as IPU) is enabled in the system. You may need to manually enable the NPU:
   ```Advanced → CPU Configuration → IPU``` 

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu.

# Detailed Getting Started Guides and Documentation: 

[Getting Started on a Versal™ board](docs/Building.md)

[Running on a Versal™ board](docs/Platform.md)

[Getting Started and Running on Windows Ryzen™ AI](docs/buildHostWin.md)

[Getting Started and Running on Linux Ryzen™ AI](docs/buildHostLin.md)

[IRON AIE Application Programming Guide](programming_guide)

[MLIR Dialect and Compiler Documentation](https://xilinx.github.io/mlir-aie/)

Interested in contributing MLIR-AIE? [Information for developers](./CONTRIBUTING.md)

-----

<p align="center">Copyright&copy; 2019-2024 Advanced Micro Devices, Inc</p>
