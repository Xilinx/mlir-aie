# Linux Build Instructions: </h1>
- Rely on Ubuntu 22.04 LTS for Vitis tool install and to build and run our MLIR-AIE tools

## Prerequisites
### MLIR-AIE tools: Ubuntu 22.04

1. Prepare your Ubuntu 22.04 machine
    - Prepare the BIOS:
        * Turn off SecureBoot (Allows for unsigned drivers to be installed)
        * BIOS → Security → Secure boot → Disable
    - Install packages (after apt-get update):
      ``` 
        flex bison libssl-dev libelf-dev
        libboost-all-dev
        libpython3.10-dev
        libsystemd-dev 
        libtiff-dev 
        libudev-dev 
      ```

1. Update the Linux kernel and install the XDNA driver and XRT:
    ```
    # Clone the XDNA driver repository:
    git clone https://github.com/amd/xdna-driver.git
    # Follow the instructions in the repository to build the kernel, driver and XRT

    # First install the iommu_sva_v4_v6.7-rc8 version of the Linux kernel:
    sudo dpkg -i linux-headers-6.7.0-rc8+_6.7.0-rc8-gf7c539200359-20_amd64.deb
    sudo dpkg -i linux-image-6.7.0-rc8+_6.7.0-rc8-gf7c539200359-20_amd64.deb 
    sudo dpkg -i linux-libc-dev_6.7.0-rc8-gf7c539200359-20_amd64.deb

    # Next reboot your machine. Once booted install xrt:
    sudo dpkg -i xrt_202410.2.17.0_22.04-amd64-xrt.deb

    # Then install the xrt_plugin:
    sudo dpkg -i xrt_plugin.2.17.0_ubuntu22.04-x86_64-amdxdna.deb

    # Check that the NPU is working if the device appears with xbutil:
    source /opt/xilinx/xrt/setup.sh
    xbutil examine

    # At the bottom of the output you should see:
        Devices present
        BDF             :  Name             
        ------------------------------------
        [0000:66:00.1]  :  RyzenAI-Phoenix 
    ```

1. Prepare Ubuntu 22.04 to build mlir-aie:
    - Install packages (after apt-get update):
      ``` 
        build-essential clang-14 lld-14 cmake
        python3-venv python3-pip
        libxrender1 libxtst6 libxi6
      ```

1. Install Vitis under from [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) and setup a AI Engine license:
    
    - Setup your environment in the following order for aietools and Vitis:
      ```
      export PATH=$PATH:<Vitis_install_path>/Vitis/2023.2/aietools/bin:<Vitis_install_path>/Vitis/2023.2/bin
      ```
    - Get local license for AIE Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense) 
    - copy license file (Xilinx.lic) to your preferred location (licenseFilePath) and update your setup configuration accordingly, for instance
      ```
      export LM_LICENSE_FILE=<licenseFilePath>/Xilinx.lic
      ```

### Quick setup for Ryzen AI application development:

1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) best under /home/username for speed (yourPathToBuildMLIR-AIE): 
   ```
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ````

1. Source `utils/quick_setup.sh` to setup the prerequisites and
   install the mlir-aie and llvm compiler tools from whls.

1. Choose a Ryzen AI reference design.
   ```
   cd add_one_objFifo
   ```

1. Jump ahead to [Build device AIE part](###Build device AIE part) step 2.

### Build mlir-aie tools from source for development:

1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) best under /home/username for speed (yourPathToBuildMLIR-AIE), with submodules: 
   ```
   git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
   ````

1. Follow regular getting started instructions [Building on x86](https://xilinx.github.io/mlir-aie/Building.html) from step 2. Please disregard any instructions referencing alternative LibXAIE versions or sysroots.

## Build a Design

For your design of interest, for instance [add_one_objFifo](../reference_designs/ipu-xrt/add_one_objFifo/), 2 steps are needed: (i) build the AIE desgin and then (ii) build the host code.

### Build device AIE part:
1. Prepare your enviroment with the mlir-aie tools (build during Prerequisites part of this guide)

    ```
    cd <yourPathToBuildMLIR-AIE>
    source <yourPathToBuildMLIR-AIE>/sandbox/bin/activate
    source yourVitisSetupScript (example shown above)
    source /opt/xilinx/xrt/setup.sh
    source <yourPathToBuildMLIR-AIE>/utils/env_setup.sh <yourPathToBuildMLIR-AIE>/install <yourPathToBuildMLIR-AIE>/llvm/install
    ```
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
    1. Alternatively, you can `sudo chown -R $USER /lib/firmware/amdipu/1502/` and remove the check for root in `/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh` (look for `!!! Please run as root !!!`).

### Build and run host part:

Note that your design of interest might need an adapted CMakelists.txt file. Also pay attention to accurately set the paths CMake parameters BOOST_ROOT, XRT_INC_DIR and XRT_LIB_DIR used in the CMakelists.txt, either in the file or as CMake command line parameters.

1. Build: Goto the same design of interest folder where the AIE design just got build (see above)
    ```
    make <testName>.exe
    ```
    
1. Run (program arguments are just an example for add_one design)
    ```
    cd Release
    .\<testName>.exe -x ..\..\build\final.xclbin -k MLIR_AIE -i ..\..\build\insts.txt -v 1
    ```

### FAQ Resetting the NPU:

    It is possible to hang the NPU in an unstable state. To reset the NPU:
    ```
    sudo rmmod amdxdna.ko
    sudo insmod <pathToYourXDNA-DriverRepo>/Release/bins/driver/amdxdna.ko
    ```

-----

<p align="center">Copyright&copy; 2019-2024 AMD</p>
