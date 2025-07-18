[![Build and Test across Python versions](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestPythons.yml/badge.svg)](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestPythons.yml) [![Build and Test with AIE tools on Ryzen™ AI](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestRyzenAI.yml/badge.svg)](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestRyzenAI.yml) [![Compile across platforms](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestMulti.yml/badge.svg)](https://github.com/Xilinx/mlir-aie/actions/workflows/buildAndTestMulti.yml)

# IRON API and MLIR-based AI Engine Toolchain

[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/mlir-aie)](https://github.com/Xilinx/mlir-aie/pulls) [![GitHub Issues](https://img.shields.io/github/issues/Xilinx/mlir-aie/bug)](https://github.com/Xilinx/mlir-aie/issues?q=is%3Aopen+is%3Aissue+label%3Abug) ![GitHub Downloads (all assets, specific tag)](https://img.shields.io/github/downloads/Xilinx/mlir-aie/latest-wheels/total?color=blue) ![GitHub Contributors](https://img.shields.io/github/contributors/Xilinx/mlir-aie)

<p align="left">
  <img src="https://github.com/llvm/mlir-www/blob/main/website/static/LogoAssets/logo/PNG/full_color/mlir-identity-03.png" alt="MLIR logo" height="80" />
  <img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png" alt="Python logo" height="80" />
  <img src="https://em-content.zobj.net/source/apple/271/mechanical-arm_1f9be.png" alt="Mechanical Arm" height="80" />
</p>

This project emphasizes fast, open-source toolchains for NPU devices including LLVM-based code generation. IRON contains a close-to-metal toolkit that empowers performance engineers to create fast and efficient designs for Ryzen™ AI NPUs powered by AI Engines. It provides Python APIs that enable developers to harness the unique architectural capabilities of AMD’s NPUs. However, this project is not intended to represent an end-to-end compilation flow for all application designs---it is designed to complement, not replace, mainstream NPU tooling for inference like the [AMD Ryzen™ AI Software Platform](https://github.com/amd/RyzenAI-SW/). Targeting researchers and enthusiasts, IRON is designed to unlock the full potential of NPUs for a wide range of workloads, from machine learning to digital signal processing and beyond. This repository includes programming guides and examples demonstrating the APIs. Additionally, the [Peano](https://github.com/Xilinx/llvm-aie) component extends the LLVM framework by adding support for the AI Engine processor as a target architecture, enabling integration with popular compiler frontends such as `clang`. Developers can leverage the [AIE API header library](https://xilinx.github.io/aie_api/topics.html) to implement efficient vectorized AIE core code in C++ that can be compiled by Peano.

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for AI Engine-enabled devices, such as [AMD Ryzen™ AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) and [Versal™](https://www.xilinx.com/products/technology/ai-engine.html).  This repository can be used to generate low-level configurations for the AI Engine portion of these devices. AI Engines are organized as a spatial array of tiles, where each tile contains AI Engine cores and/or memories. The spatial array is connected by stream switches that can be configured to route data between AI Engine tiles scheduled by their programmable Data Movement Accelerators (DMAs). This repository contains MLIR representations, with multiple levels of abstraction, to target AI Engine devices. This enables compilers and developers to program AI Engine cores, as well as describe data movements and array connectivity.

The IRON Python API for Ryzen™ AI NPUs is described in the following paper: 

> E. Hunhoff, J. Melber, K. Denolf, A. Bisca, S. Bayliss, S. Neuendorffer, J. Fifield, J. Lo, P. Vasireddy, P. James-Roxby, E. Keller. "[Efficiency, Expressivity, and Extensibility in a Close-to-Metal NPU Programming Interface](https://arxiv.org/abs/2504.18430)". In 33rd IEEE International Symposium On Field-Programmable Custom Computing Machines, May 2025.

# Getting Started for AMD Ryzen™ AI on Linux

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 24.04** or **Ubuntu 24.10** install.

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

### Install IRON and MLIR-AIE Prerequisites

1. Install the following packages needed for MLIR-AIE:

    ```bash
    # Python versions 3.10, 3.12 and 3.13 are currently supported by our wheels
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip
    ```

1. (Optional) Install opencv which is needed for vision programming examples:

   ```bash
   sudo apt install libopencv-dev python3-opencv
   ```

## Install IRON for AMD Ryzen™ AI AIE Application Development

1. Clone [the mlir-aie repository](https://github.com/Xilinx/mlir-aie.git):
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ```

1. Setup a virtual environment:
   ```bash
   python3 -m venv ironenv
   source ironenv/bin/activate
   python3 -m pip install --upgrade pip
   ```

1. Install IRON library, mlir-aie and llvm-aie compilers from whls:

   For release v1.0:
   ```bash
   # Install IRON library and mlir-aie from a wheel
   python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0

   # Install Peano from a llvm-aie wheel
   python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
   ```

   For daily latest:
   ```bash
   # Install IRON library and mlir-aie from a wheel
   python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2

   # Install Peano from llvm-aie wheel
   python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
   ```

1. Install required Python packages:
   ```bash
   # Install basic Python requirements (still needed for release v1.0, but is no longer needed for latest wheels)
   python3 -m pip install -r python/requirements.txt

   # This installs the pre-commit hooks defined in .pre-commit-config.yaml
   pre-commit install

   # Install MLIR Python Extras 
   HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
   ```

1. (Optional) Install ML Python packages for ml programming examples:
   ```bash
   # Install Torch for ML examples
   python3 -m pip install -r python/requirements_ml.txt
   ```

1. (Optional) Install Jupyter Notebook Python packages:
   ```bash
   # This creates an ipykernel (for use in notebooks) using the ironenv venv
   python3 -m ipykernel install --user --name ironenv
    
   # The install generally captures in the $PYTHONPATH by the `env_setup.sh` script.
   # However, jupyter notebooks don't always get access to the PYTHONPATH (e.g. if they are run with
   # vscode) so we save the ${MLIR_AIE_INSTALL_DIR}/python in a .pth file in the site packages dir of the
   # ironenv venv; this allows the iron ipykernel to find the install dir regardless of if PYTHONPATH is
   # available or not.
   MLIR_AIE_INSTALL=`$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie` \
   venv_site_packages=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'` \
   echo ${MLIR_AIE_INSTALL}/python > $venv_site_packages/mlir-aie.pth
   ```

1. Setup environment and add tools to PATHs
   ```bash
   source utils/env_setup.sh
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

## Learn more about NPU programming with IRON

1. Continue to the [IRON AIE Application Programming Guide](programming_guide)

1. Additional MLIR-AIE documentation is available on the [website](https://xilinx.github.io/mlir-aie/)

1. AIE API header library documentation for single-core AIE programming in C++ is avaiable [here](https://xilinx.github.io/aie_api/topics.html)

1. If you are a university researcher or student and interested in trying these tools on our Ryzen™ AI AUP Cloud systems, please contact the [AMD University Program](mailto:aup@amd.com)

## Optional: Install AIETools

> You may skip the Vitis™ installation step if you intend to only target AMD XDNA™/AIE-ML (AIE2) and AMD XDNA™ 2 (AIE2P) using our open-source single-core compiler [Peano](https://github.com/Xilinx/llvm-aie). Compiling with `xchesscc` is not supported without installing AMD Vitis™ AIE Essentials. 

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

## Troubleshooting:

### Update BIOS:

Be sure you have the latest BIOS for your laptop or mini PC, this will ensure the NPU (sometimes referred to as IPU) is enabled in the system. You may need to manually enable the NPU:
   ```Advanced → CPU Configuration → IPU``` 

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu.

# Detailed Getting Started Guides and Documentation: 

[IRON AIE Application Programming Guide](programming_guide)

[Building mlir-aie tools from source](docs/Building.md)

[MLIR Dialect and Compiler Documentation](https://xilinx.github.io/mlir-aie/)

Interested in contributing MLIR-AIE? [Information for developers](./CONTRIBUTING.md)

-----

<p align="center">Copyright&copy; 2019-2024 Advanced Micro Devices, Inc</p>
