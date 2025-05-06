# Windows Setup and Build Instructions

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU on Windows. The instructions were tested on a ASUS Vivobook Pro 15 among other Phoenix and Strix equipped machines.

You will set up a Windows subsystem for Linux (WSL) Ubuntu install, which will be used for building NPU device code. For building the host (x86) code, you will use MS Visual Code Community but this can be invoked from within WSL so you should be able to compile and run code entirely within WSL. 

- Rely on WSL Ubuntu 22.04 LTS for tool install and to build and run our mlir-aie tools
- Rely on MS Visual Studio 17 2022 to natively build the host code (aka test.cpp) but this can be invoked from within WSL


## Setup and Build Steps
1. [Prerequisites](#prerequisites) - Clone mlir-aie from within WSL, install dependent sw packages, Clone/unzip XRT (no need to build), 
1. [Prepare Host Side: Natively on Win11](#prepare-host-side:-natively-on-win11)
1. [Setup up your environment](#set-up-your-environment)
1. [Build a Design](#build-a-design)
* [Optional: Install AIETools](#optinal:-install-aietools)
* [Troubleshooting](#Troubleshooting)


## Prerequisites
### mlir-aie tools: WSL Ubuntu 22.04
All steps in WSL Ubuntu terminal.

1. Prepare WSL2 with Ubuntu 22.04:
    - Install packages (after apt-get update):
      ```
        sudo apt install \
        build-essential clang clang-14 lld lld-14 cmake \
        python3-venv python3-pip \
        libxrender1 libxtst6 libxi6 \
        mingw-w64-tools
      ```
    - generate locales
      ```
      apt-get install locales
      locale-gen en_US.UTF-8
      ```

1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) best under /home/username for speed (yourPathToBuildMLIR-AIE), with submodules:
   ```
   git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
   ````
   This github repo will be used for building the IRON tools as cloning/ updating/ building in WSL is much faster for files stored in WSL drive space. A separate clone of the mlir-aie repo will be used for working with example designs so that powershell can find the diretory path. See [below](#prepare-host-side:-natively-on-win11)
   
1. Install or Build mlir-aie tools under WSL2:

   * Use quick setup script to install from whls:

     >  NOTE: Installing the mlir-aie tools from wheels via the quick setup path supports AMD XDNA™/AIE-ML (AIE2) and AMD XDNA™ 2 (AIE2P), it does NOT support Versal™ devices with AIE.

     ```
     source utils/quick_setup.sh
     # NOTE: this will install mlir-aie in my_install/mlir_aie
     # Be sure to account for this using utils/env_setup.sh later on.
     ```

   * [Optional] Build from source following regular get started instructions [https://xilinx.github.io/mlir-aie/Building.html](https://xilinx.github.io/mlir-aie/Building.html)

1. After installing the updated Ryzen™ AI driver (see next subsection), use the gendef tool (from the mingw-w64-tools package) to create a .def file with the symbols. This step is needed to create an XRT .dll file that we can link against when we compile. 
    ```
    mkdir /mnt/c/Technical/xrtNPUfromDLL; cd /mnt/c/Technical/xrtNPUfromDLL
    cp /mnt/c/Windows/System32/AMD/xrt_coreutil.dll .
    gendef xrt_coreutil.dll
    ```

1. Clone XRT under `C:\Technical\XRT`. 

### Prepare Host Side: Natively on Win11

All steps in Win11 (powershell where needed).

1. Upgrade the NPU driver to version 10.106.8.62 [download here](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ipu_stack_rel_silicon_2308.zip), following the [instructions](href="https://ryzenai.docs.amd.com/en/latest/inst.html) on setting up the driver. Note that we currently have two steps for driver update. This version provides the `xrt_coreutil.dll` under `C:\Windows\System32\AMD` which is needed to generate the `xrt_coreutil.lib`. However, we also want to install the most up-to-date NPU driver package linked from [here](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers) under `NPU Driver`. Use version 10.106.8.62 to generate the `xrt_coreutil.lib`, then come back and upgrade the driver to the most up-to-date one.

1. Install [Microsoft Visual Studio 17 2022 Community Edition](https://visualstudio.microsoft.com/vs/community/) with package for C++ development.

1. Install CMake on windows ([https://cmake.org/download/](https://cmake.org/download/))
1. Optional (only needed for vision examples): install [opencv](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html) and add this install to your PATH environmental variable, for instance `C:\Technical\thirdParty\opencv\build\x64\vc16\bin`

1. Clone [https://github.com/Xilinx/XRT](https://github.com/Xilinx/XRT) for instance under `C:\Technical` and `git checkout 2024.2`
1. Create a .lib file from the .dll shipping with the driver
    - In wsl, generate a .def file (see above)
    - Start a x86 Native Tools Command Prompt (installed as part of VS17), go to the folder `C:\Technical\xrtNPUfromDLL` and run command:
      ```
      lib /def:xrt_coreutil.def /machine:x64 /out:xrt_coreutil.lib
      ```
1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) for instance under `C:\Technical` to be used to build your designs (yourPathToDesignsWithMLIR-AIE). This is needed because Visual Studio Compiler needs standard Windows directory paths rather than WSL mounted paths to work properly.

## Set up your environment

To make the compilation toolchain available for use in your WSL terminal, you will need to set some environment variables. We suggest you add the following to a file named `setup.sh`, so you can set up your environment easily by running `source setup.sh`.

### `setup.sh` - Option A - Using Quick Setup

If you used the quick setup script (precompiled mlir-aie binaries), use this setup script.

```
# NOTE: if you did NOT exit the terminal you can skip this step.
cd <yourPathToDesignsWithMLIR-AIE>
source <yourPathToBuildMLIR-AIE>/ironenv/bin/activate
source yourVitisSetupScript (example shown above)
source <yourPathToBuildMLIR-AIE>/utils/env_setup.sh <yourPathToBuildMLIR-AIE>/my_install/mlir_aie
```

### `setup.sh` - Option B - Built from Source

```
cd <yourPathToDesignsWithMLIR-AIE>
source <yourPathToBuildMLIR-AIE>/sandbox/bin/activate
source yourVitisSetupScript (example shown above)
source <yourPathToBuildMLIR-AIE>/utils/env_setup.sh <yourPathToBuildMLIR-AIE>/install
```


## Build a Design

For your design of interest, for instance from [programming_examples](../programming_examples/), 2 steps are needed: (i) build the AIE desgin in WSL and then (ii) build the host code in powershell. The host code build step can be invoked from within WSL, leveraging Makefile variables `$powershell` and `$getwslpath` from [programming_examples/makefile-common](../programming_examples/makefile-common). See [Makefile](../programming_examples/basic/vector_scalar_mul/Makefile) for an example of how it is used.

### Build device AIE part: WSL Ubuntu terminal
1. Prepare your enviroment with the mlir-aie tools (built during Prerequisites part of this guide). See [Set up your environment](#set-up-your-environment) above.

1. Goto the design of interest such as [vector_scalar_mul](../programming_examples/basic/vector_scalar_mul/) and run `make run`.


## Optional: Install AIETools

> You may skip the Vitis™ installation step if you intend to only target AMD XDNA™/AIE-ML (AIE2) and AMD XDNA™ 2 (AIE2P) using our open-source single-core compiler [Peano](https://github.com/Xilinx/llvm-aie). Compiling with `xchesscc` is not supported without installing AMD Vitis™ AIE Essentials. 

1. Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Access](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab). We will assume you use the installation directory, `/tools/ryzen_ai-1.3.0/vitis_aie_essentials` from within WSL.

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

    1. Get a local license for AI Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense) providing your machine's MAC address (`ip -brief link show eth0`). Be sure to select License Type of `Node` instead of `Floating`.

    1. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic` where `<licenseFilePath>` is `/opt`, then update your setup configuration accordingly, for instrance:
        ```
        export XILINXD_LICENSE_FILE=<licenseFilePath>/Xilinx.lic
        ip link add vmnic0 type dummy
        ip link set vmnic0 addr <yourMACaddress>
        ```
       
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

#### Other Helpful BIOS Settings:
1. Turn off SecureBoot (Allows for unsigned drivers to be installed)

   ```BIOS → Security → Secure boot → Disable```

1. Turn Ac Power Loss to "Always On" (Can be used for PDU reset, turns computer back on after power loss)

   ```BIOS → Advanced → AMD CBS →  FCH Common Options → Ac Power Loss Options → Set Ac Power Loss to "Always On"```

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu.

-----

<p align="center">Copyright&copy; 2019-2024 Advanced Micro Devices, Inc</p>
