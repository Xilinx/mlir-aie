# Linux Setup and Build Instructions

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 24.10** install with Linux 6.11 kernel. 

## Initial Setup

#### Update BIOS:

Be sure you have the latest BIOS for your laptop or mini PC, this will ensure the NPU (sometimes referred to as IPU) is enabled in the system. You may need to manually enable the NPU:
   ```Advanced → CPU Configuration → IPU``` 

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu.

#### BIOS Settings:

Turn off SecureBoot (Allows for unsigned drivers to be installed):
   ```BIOS → Security → Secure boot → Disable```

## Prerequisites

### Install Xilinx Vitis 2023.2 

1. Install Vitis under from [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html). You will need to run the installer as root. We will assume you use the default installation directory, `/tools/Xilinx`.

   > This is a large download. A wired connection will speed things up. Be prepared to spend multiple hours on this step.

1. Set up a AI Engine license.

    1. Get a local license for AIE Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).

    1. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`:
       
    1. Setup your environment using the following script for Vitis for aietools:

       ```bash
       #!/bin/bash
        #################################################################################
        # Setup Vitis (which is just for aietools)
        #################################################################################
        export MYXILINX_VER=2023.2
        export MYXILINX_BASE=/tools/Xilinx
        export XILINX_LOC=$MYXILINX_BASE/Vitis/$MYXILINX_VER
        export AIETOOLS_ROOT=$XILINX_LOC/aietools
        export PATH=$PATH:${AIETOOLS_ROOT}/bin
        export LM_LICENSE_FILE=/opt/Xilinx.lic
       ```
   1. Vitis requires some python3.8 libraries:
  
      ```bash
      sudo add-apt-repository ppa:deadsnakes/ppa
      sudo apt-get update
      sudo apt install libpython3.8-dev
      ```

### Install the XDNA Driver

1. Install the following prerequisite packages.
 
   ```bash
   sudo apt install \
   libidn11-dev
   ```

1. Clone the XDNA driver repository and its submodules.
    ```bash
    git clone https://github.com/amd/xdna-driver.git
    export XDNA_SRC_DIR=$(realpath xdna-driver)
    cd xdna-driver
    git reset --hard 3d5a8cf1af2adfbb6306ad71b45e5f3e1ffc5b37
    git submodule update --init --recursive
    ```

    > The submodules use SSH remotes. You will need a GitHub account and locally installed SSH keys to pull the submodules. Follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to set up an SSH key. Alternatively, edit `.gitmodules` to use HTTPS instead of SSH.

1. Install XRT. (Below steps are adapted from [here](https://xilinx.github.io/XRT/master/html/build.html).)

    1. Install XRT prerequisites.
    
       ```bash
       cd $XDNA_SRC_DIR
       sudo ./tools/amdxdna_deps.sh
       ```

    2. Build XRT. Remember to source the aietools/Vitis setup script from [above](#install-xilinx-vitis-20232).

       ```bash
       cd $XDNA_SRC_DIR/xrt/build
       ./build.sh -noert -noalveo
       ```

    3. Install XRT.

       ```bash
       cd $XDNA_SRC_DIR/xrt/build/Release
       sudo apt reinstall ./xrt_202420.2.18.0_24.10-amd64-xrt.deb ./xrt_202420.2.18.0_24.10-amd64-xbflash.deb
       ```

       > **An error is expected in this step.** Ignore it.



1. Build XDNA-Driver. Below steps are adapted from [here](https://github.com/amd/xdna-driver).

    ```bash
    cd $XDNA_SRC_DIR/build
    ./build.sh -release
    ./build.sh -package
    ```

1. Install XDNA.

    ```bash
    cd $XDNA_SRC_DIR/build/Release
    sudo apt reinstall ./xrt_plugin.2.18.0_ubuntu24.10-x86_64-amdxdna.deb
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
    build-essential clang clang-14 lld lld-14 cmake python3-venv python3-pip libxrender1 libxtst6 libxi6 virtualenv
    ```

1. Install g++13 and opencv needed for some programming examples:

   ```bash
   sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   sudo apt update
   sudo apt install gcc-13 g++-13 -y
   sudo apt install libopencv-dev python3-opencv
   ```

1. Remember to source the aietools/Vitis setup script from [above](#install-xilinx-vitis-20232).

1. Remeber to source the xrt setup script: `source /opt/xilinx/xrt/setup.sh`

### Install IRON for Ryzen™ AI AIE Application Development

1. Clone [the mlir-aie repository](https://github.com/Xilinx/mlir-aie.git), best under /home/username for speed (yourPathToBuildMLIR-AIE): 
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ```

1. Source `utils/quick_setup.sh` to setup the prerequisites and
   install the mlir-aie and llvm compiler tools from whls.

## Build a Design

> Remember to set up your environment including Vitis, your license, XRT, and IRON
> ```
>   source yourVitisSetupScript.sh
>   export LM_LICENSE_FILE=/opt/Xilinx.lic
>   source /opt/xilinx/xrt/setup.sh
>   source utils/setup_iron_env.sh
> ```

For your design of interest, for instance from [programming_examples](../programming_examples/), 2 steps are needed: (i) build the AIE desgin and then (ii) build the host code.

### Build Device AIE Part

1. Goto the design of interest and run `make`

### Build and Run Host Part

1. Build: Goto the same design of interest folder where the AIE design just got built (see above)
    ```bash
    make <testName>.exe
    ```
    > Note that the host code target has a `.exe` file extension even on Linux. Although unusual, this is an easy way for us to distinguish whether we want to compile device code or host code.


1. Run (program arguments are just an example for add_one design)
    ```bash
    cd Release
    .\<testName>.exe -x ..\..\build\final.xclbin -k MLIR_AIE -i ..\..\build\insts.txt -v 1
    ```

# Troubleshooting

## License Errors When Trying to Compile

The `v++` compiler for the NPU device code requires a valid Vitis license. If you are getting errors related to this:

1. You have obtained a valid license, as described [above](#install-xilinx-vitis-20232-and-other-mlir-aie-prerequisites). 
1. Make sure you have set the environment variable `LM_LICENSE_FILE` to point to your license file, see [above](#setting-up-your-environment).
1. Make sure the ethernet interface whose MAC address you used to generate the license is still available on your machine. For example, if you used the MAC address of a removable USB Ethernet adapter, and then removed that adapter, the license check will fail. You can list MAC addresses of interfaces on your machine using `ip link`.

-----

<p align="center">Copyright&copy; 2024 AMD</p>
