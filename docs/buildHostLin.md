# Linux Setup and Build Instructions

These instructions will guide you through everything required for building and executing a program on the Ryzen AI NPU, starting from a fresh bare-bones **Ubuntu 22.04 LTS** install. Only Ubuntu 22.04 LTS is supported. The instructions were tested on a ASUS Vivobook Pro 15. 

## Overview
You will...

1. Install a driver for the Ryzen AI. As part of this, you will need to...

   1. [...compile and install a more recent Linux kernel.](#update-linux)

   1. [...compile and install the XDNA driver from source.](#install-the-xdna-driver)

1. Install the compiler toolchain, allowing you to compile your own NPU designs from source. As part of this, you will need to...

   1. [...install Xilinx Vitis and obtain a license.](#install-xilinx-vitis-20232-and-other-mlir-aie-prerequisites)

   1. ...install MLIR-AIE [from precompiled binaries (fast)](#option-a---quick-setup-for-ryzen-ai-application-development) or [from source (slow)](#option-b---build-mlir-aie-tools-from-source-for-development).

1. Build and execute one of the example designs. This consists of...

   1. [...setting up your environment.](#setting-up-your-environment)
   
   2. [...building device (NPU) code.](#build-device-aie-part)
   
   3. [...building and executing host (x86) code and device (NPU) code.](#build-and-run-host-part) 

> Be advised that two of the steps (Linux compilation and Vitis install) may take hours. If you decide to build MLIR-AIE from source, this will also take a long time as it contains an LLVM build. Allocate enough time and patience. Once done, you will have an amazing toolchain allowing you to harness this great hardware at your hands.

## Prerequisites

### Update Linux

> The reason we need to update the kernel is that the XDNA driver requires IOMMU SVA support.

1. Disable **Secure Boot** in the BIOS. This allows for unsigned drivers to be installed.

    >  On the ASUS Vivobook, this setting can be found under
      BIOS → Advanced Settings (F7) → Security →  Secure Boot → Secure Boot Control (Set to Disabled)

1. Install the following prerequisite packages for compiling Linux:
    ```
    sudo apt install \
    build-essential debhelper flex bison libssl-dev libelf-dev libboost-all-dev libpython3.10-dev libsystemd-dev libtiff-dev libudev-dev
    ```

1. Pull the source for the correct kernel version, which is available in the AMDESE linux repository.

    ```
    git clone --branch iommu_sva_v4_v6.7-rc8 https://github.com/AMDESE/linux.git
    export LINUX_SRC_DIR=$(realpath linux)
    ```

1. Create a build directory and a configuration within it.
    
    ```
    mkdir linux-build
    export LINUX_BUILD_DIR=$(realpath linux-build)
    cp /boot/config-`uname -r` $LINUX_BUILD_DIR/.config
    ```

1. Go to the directory where you cloned Linux and adjust the configuration.

    ```
    cd $LINUX_SRC_DIR
    make olddefconfig
    ./scripts/config --file $LINUX_BUILD_DIR/.config --disable MODULE_SIG
    ./scripts/config --file $LINUX_BUILD_DIR/.config --enable DRM_ACCEL
    ```

1. Build Linux.

    ```
    make -j$(nproc) O=$LINUX_BUILD_DIR bindeb-pkg 2>&1 | tee kernel-build.log
    ```

    > Compiling the linux kernel may take hours.
    
    > Note that the final kernel `.deb` packages will be in the *parent* directory of `LINUX_BUILD_DIR`.

1. Install the new Linux kernel and reboot.

    ```
    cd $LINUX_BUILD_DIR/..
    sudo dpkg -i linux-headers-6.7.0-rc8+_6.7.0-rc8-gf7c539200359-20_amd64.deb
    sudo dpkg -i linux-image-6.7.0-rc8+_6.7.0-rc8-gf7c539200359-20_amd64.deb 
    sudo dpkg -i linux-libc-dev_6.7.0-rc8-gf7c539200359-20_amd64.deb
    sudo shutdown --reboot 0
    ```

### Install the XDNA Driver

1. Install a more recent CMake, which is needed for building XRT.
   
   1. Download CMake 3.28 binaries into `NEW_CMAKE_DIR`.
      ```
      mkdir cmake
      export NEW_CMAKE_DIR=$(realpath cmake)
      cd cmake
      wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh
      chmod +x ./cmake-3.28.3-linux-x86_64.sh
      ./cmake-3.28.3-linux-x86_64.sh
      ```

   1. Answer the prompts with **y** (accept license), then **n** (include subdirectory).

   1. Add new cmake directory to your `PATH`.

      ```
      export PATH="${NEW_CMAKE_DIR}/bin":"${PATH}"
      ```
   
   1. Verify the install of CMake was successful.

      ```
      cmake --version
      ```

      > The frist line this prints should read
      > ```cmake version 3.28.3```

1. Install the following prerequisite packages.
 
   ```
   sudo apt install \
   libidn11-dev
   ```

1. Clone the XDNA driver repository and its submodules.
    ```
    git clone https://github.com/amd/xdna-driver.git
    export XDNA_SRC_DIR=$(realpath xdna-driver)
    cd xdna-driver
    git reset --hard 317e0c67747cbf88e5b5a3a81ba4bdf7bf5b3fc3
    git submodule update --init --recursive
    ```

    > The submodules use SSH remotes. You will need a GitHub account and locally installed SSH keys to pull the submodules. Follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to set up an SSH key. Alternatively, edit `.gitmodules` to use HTTPS instead of SSH.

1. Install XRT. (Below steps are adapted from [here](https://xilinx.github.io/XRT/master/html/build.html).)

    1. Install XRT prerequisites.
    
       ```
       cd $XDNA_SRC_DIR/xrt
       sudo ./runtime_src/tools/scripts/xrtdeps.sh
       ```

    2. Build XRT.

       ```
       cd $XDNA_SRC_DIR/xrt/build
       ./build.sh
       cd Release
       make package
       ```

    3. Install XRT.

       ```
       cd $XDNA_SRC_DIR/xrt/build
       sudo dpkg -i xrt_202410.2.17.0_22.04-amd64-xrt.deb
       ```

       > **An error is expected in this step.** Ignore it.

1. Install the XDNA prerequisites. (Below steps are adapted from [here](https://github.com/amd/xdna-driver).)

    ```
    cd $XDNA_SRC_DIR
    sudo su
    ./tools/amdxdna_deps.sh
    exit
    ```

1. Build XDNA.

    ```
    cd $XDNA_SRC_DIR/build
    ./build.sh -release
    ./build.sh -package
    ```

1. Install XDNA.

    ```
    cd $XDNA_SRC_DIR/build
    sudo dpkg -i xrt_plugin.2.17.0_ubuntu22.04-x86_64-amdxdna.deb
    ```
    
1. Check that the NPU is working if the device appears with xbutil:
   
   ```
   source /opt/xilinx/xrt/setup.sh
   xbutil examine
   ```

   > At the bottom of the output you should see:
   >  ```
   >  Devices present
   >  BDF             :  Name             
   > ------------------------------------
   >  [0000:66:00.1]  :  RyzenAI-Phoenix 
   >  ```

### Install Xilinx Vitis 2023.2 and Other MLIR-AIE Prerequisites

1. Install Vitis under from [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html). You will need to run the installer as root. We will assume you use the default installation directory, `/tools/Xilinx`.

   > This is a large download. A wired connection will speed things up. Be prepared to spend multiple hours on this step.

1. Set up a AI Engine license.
    
    1. Setup your environment in the following order for aietools and Vitis:

       ```
       source /tools/Xilinx/Vitis/2023.2/settings64.sh
       ```

    1. Get a local license for AIE Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).

    1. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`, and set the `LM_LICENSE_FILE` environment variable:

       ```
       export LM_LICENSE_FILE=/opt/Xilinx.lic
       ```

1. Install the following packages needed for building MLIR-AIE:
    ``` 
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake python3-venv python3-pip libxrender1 libxtst6 libxi6
      ```

1. Choose *one* of the two options (A or B) below for installing MLIR-AIE.

### Option A - Quick Setup for Ryzen AI Application Development

1. Clone [the MLIR-AIE repository](https://github.com/Xilinx/mlir-aie.git), best under /home/username for speed (yourPathToBuildMLIR-AIE): 
   ```
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ````

1. Source `utils/quick_setup.sh` to setup the prerequisites and
   install the mlir-aie and llvm compiler tools from whls.

1. Jump ahead to [Build Device AIE Part](#build-device-aie-part) step 2 below.

### Option B - Build MLIR-AIE Tools from Source for Development

1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) best under /home/username for speed (yourPathToBuildMLIR-AIE), with submodules: 
   ```
   git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
   ````

1. Follow regular getting started instructions [Building on x86](https://xilinx.github.io/mlir-aie/Building.html) from step 2. Please disregard any instructions referencing alternative LibXAIE versions or sysroots.

## Setting up your Environment

After all prerequisites (drivers and compilation toolchain) have been installed, you need to make them findable by adding them to the `PATH` and setting required environment variables.

We suggest you add all of the following to a `setup.sh` script in your home directory, and `source setup.sh` as the first step of your workflow. That way, everything is set up in one setp.


### `setup.sh` - Option A - Quick Setup

```
export LM_LICENSE_FILE=/opt/Xilinx.lic
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export PATH="${NEW_CMAKE_DIR}/bin":"${PATH}"

cd ${MLIR_AIE_BUILD_DIR}
source ${MLIR_AIE_BUILD_DIR}/ironenv/bin/activate
source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh ${MLIR_AIE_BUILD_DIR}/my_install/mlir_aie ${MLIR_AIE_BUILD_DIR}/my_install/mlir
```

> Replace `${MLIR_AIE_BUILD_DIR}` with the directory in which you *built* MLIR-AIE above. Replace `${NEW_CMAKE_DIR}` with the directory in which you installed CMake 3.28 above. Instead of search and replace, you can also define these values as environment variables.

> For quick setup, this step is only needed if you are starting with a new terminal. If you are continuing in the same terminal you used to install the prerequisites, the environment variables should all be set.

### `setup.sh` - Option B - Toolchain Compiled From Source

```
cd ${MLIR_AIE_BUILD_DIR}
source ${MLIR_AIE_BUILD_DIR}/sandbox/bin/activate
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh ${MLIR_AIE_BUILD_DIR}/install ${MLIR_AIE_BUILD_DIR}/llvm/install
```

> Replace `${MLIR_AIE_BUILD_DIR}` with the directory in which you *built* MLIR-AIE above. Instead of search and replace, you can also define `MLIR_AIE_BUILD_DIR` as an environment variable.

## Build a Design

For your design of interest, for instance [add_one_objFifo](../reference_designs/ipu-xrt/add_one_objFifo/), 2 steps are needed: (i) build the AIE desgin and then (ii) build the host code.

### Build Device AIE Part

1. Prepare your enviroment with the MLIR-AIE tools (built during prerequisites part of this guide) - see **"Setting Up Your Environment"** avove.

2. Goto the design of interest and run `make`

3. Signing your array configuration binary aka. XCLBIN
    ```
    sudo bash
    source /opt/xilinx/xrt/setup.sh
    # Assume adding an unsigned xclbin on Phoenix, run
    /opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin <your test>.xclbin

    # <your test>_unsigned.xclbin will be added into /lib/firmware/amdxdna/<version>/ and symbolic link will create.
    # When xrt_plugin package is removed, it will automatically cleanup.
    ```
    1. Alternatively, you can `sudo chown -R $USER /lib/firmware/amdnpu/1502/` and remove the check for root in `/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh` (look for `!!! Please run as root !!!`).

### Build and Run Host Part

Note that your design of interest might need an adapted `CMakeLists.txt` file. Also pay attention to accurately set the paths CMake parameters `BOOST_ROOT`, `XRT_INC_DIR` and `XRT_LIB_DIR` used in the `CMakeLists.txt`, either in the file or as CMake command line parameters.

1. Build: Goto the same design of interest folder where the AIE design just got built (see above)
    ```
    make <testName>.exe
    ```
    > Note that the host code target has a `.exe` file extension even on Linux. Although unusual, this is an easy way for us to distinguish whether we want to compile device code or host code.


1. Run (program arguments are just an example for add_one design)
    ```
    cd Release
    .\<testName>.exe -x ..\..\build\final.xclbin -k MLIR_AIE -i ..\..\build\insts.txt -v 1
    ```

# Troubleshooting

## Resetting the NPU

It is possible to hang the NPU in an unstable state. To reset the NPU:

```
sudo rmmod amdxdna.ko
sudo insmod $XDNA_SRC_DIR/build/Release/bins/driver/amdxdna.ko
```

If you installed the AMD XDNA driver using `.deb` packages as outlined above, and `insmod` does not work, you may instead want to try:

```
sudo modprobe -r amdxdna
sudo modprobe -v amdxdna
```

## `xrt_core::system_error` - Unsigned xclbins

If you are able to successfully build your design, but are getting the following error when trying to execute it:

```
terminate called after throwing an instance of 'xrt_core::system_error'
  what():  DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=2): No such file or directory
Aborted (core dumped)
```

This may be because you did not sign your `final.xclbin`. The device only allows executing signed xclbins. Follow step 3 under section [Build Device AIE Part](#build-device-aie-part) above.

## Signing the `xclbin` hangs

As outlined above, `.xclbin` files must be signed to be able to run on the device. Signing is done by running

```
/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin <your test>.xclbin
```

This may hang after the following output if you have too many signed `.xclbin`s:

```
Copy <your test>.xclbin to /lib/firmware/amdnpu/1502/<your test>.xclbin
```

If this happens, clear all your previously signed `.xclbin`s as follows (you will of course have to re-sign the ones you remove in this step if you want to run them again, but chances are you have many old unneeded `.xclbin`s in there):

```
rm /lib/firmware/amdnpu/1502/<your tests>.xclbin
```

## License Errors When Trying to Compile

The `v++` compiler for the NPU device code requires a valid Vitis license. If you are getting errors related to this:

1. You have obtained a valid license, as described [above](#install-xilinx-vitis-20232-and-other-mlir-aie-prerequisites). 
1. Make sure you have set the environment variable `LM_LICENSE_FILE` to point to your license file, see [above](#setting-up-your-environment).
1. Make sure the ethernet interface whose MAC address you used to generate the license is still available on your machine. For example, if you used the MAC address of a removable USB Ethernet adapter, and then removed that adapter, the license check will fail. You can list MAC addresses of interfaces on your machine using `ip link`.

-----

<p align="center">Copyright&copy; 2019-2024 AMD</p>
