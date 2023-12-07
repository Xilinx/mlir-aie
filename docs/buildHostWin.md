# Windows Build Instructions: WSL Ubuntu for tools, MS Visual Code Community for host test code</h1>
- Rely on WSL Ubuntu 22.04 LTS for Vitis tool install and to build and run our MLIR-AIE tools
- Rely on MS Visual Studio 17 2022 to natively build the host code (aka test.cpp)
## Prerequisites
### MLIR-AIE tools: WSL Ubuntu 22.04
All steps in WSL Ubuntu terminal.
1. Clone [https://github.com/Xilinx/mlir-aie.git](https://github.com/Xilinx/mlir-aie.git) best under /home/username for speed (yourPathToBuildMLIR-AIE), with submodules: 
   ```
   git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git
   ````
1. Prepare WSL2 with Ubuntu 22.04:
    - Install packages (after apt-get update):
      ``` 
        build-essential clang-14 lld-14 cmake
        libboost-all-dev
        python3-venv python3-pip
        libxrender1 libxtst6 libxi6
        mingw-w64-tools
      ```
    - generate locales
      ```
      apt-get install locales
      locale-gen en_US.UTF-8
      ```

1. Install Vitis under WSL Ubuntu from [Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) and setup a AI Engine license:
    
    - Setup your environment in the following order for aietools and Vitis:
      ```
      export PATH=$PATH:<Vitis_install_path>/Vitis/2023.2/aietools/bin:<Vitis_install_path>/Vitis/2023.2/bin
      ```
    - Get local license for AIE Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense) providing your machine's MAC address (`ip -brief link show eth0`) 
    - copy license file (Xilinx.lic) to your preferred location (licenseFilePath) and update your setup configuration accordingly, for instance
      ```
      export XILINXD_LICENSE_FILE=<licenseFilePath>/Xilinx.lic
      ip link add vmnic0 type dummy
      ip link set vmnic0 addr <yourMACaddress>
      ```

1. Build MLIR-AIE tools under WSL2 following regular get started instructions [https://xilinx.github.io/mlir-aie/Building.html](https://xilinx.github.io/mlir-aie/Building.html)

1. After installing the updated RyzenAI driver (see next subsection), use the gendef tool (from the mingw-w64-tools package) to create a .def file with the symbols:
    ```
    mkdir /mnt/c/Technical/xrtIPUfromDLL; cd /mnt/c/Technical/xrtIPUfromDLL
    cp /mnt/c/Windows/System32/AMD/xrt_coreutil.dll .
    gendef xrt_coreutil.dll
    ```

### Prepare Host Side: Natively on Win11

All steps in Win11 (powershell where needed).

1. Upgrade the IPU driver IPU driver to version 10.106.8.62 [download here](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ipu_stack_rel_silicon_2308.zip), following the [instructions](href="https://ryzenai.docs.amd.com/en/latest/inst.html) on setting up the driver.
1. Install [Microsoft Visual Studio 17 2022 Community Edition](https://visualstudio.microsoft.com/vs/community/) with package for C++ development.

1. Install CMake on windows ([https://cmake.org/download/](https://cmake.org/download/))
    - [Download](https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.zip) and [compile](https://www.boost.org/doc/libs/1_83_0/more/getting_started/windows.html) boost (current version 1.83). 
    - Extract zip file into `C:\Technical\thirdParty`
    - Run `bootstrap.bat` and after that `b2.exe`
1. Optional (only needed for vision examples): install [opencv](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html) and add this install to your PATH environmental variable, for instance `C:\Technical\thirdParty\opencv\build\x64\vc16\bin`

1. Clone [https://github.com/Xilinx/XRT](https://github.com/Xilinx/XRT) for instance under `C:\Technical` and `git checkout 2023.2`
1. Create a .lib file from the .dll shipping with the driver
    - In wsl, generate a .def file (see above)
    - Start a x86 Native Tools Command Prompt (installed as part of VS17), go to the folder `C:\Technical\xrtIPUfromDLL` and run command: 
      ```
      lib /def:xrt_coreutil.def /machine:x64 /out:xrt_coreutil.lib
      ```
1. Clone [https://github.com/Xilinx/mlir-aie.git]([https://gitenterprise.xilinx.com/XRLabs/pynqMLIR-AIE](https://github.com/Xilinx/mlir-aie.git)) for instance under C:\Technical to be used to build designs (yourPathToDesignsWithMLIR-AIE) 


## Build a Design

For your design of interest, for instance [add_one_objFifo](../reference_designs/ipu-xrt/add_one_objFifo/), 2 steps are needed: (i) build the AIE desgin in WSL and then (ii) build the host code in powershell.

### Build device AIE part: WSL Ubuntu terminal
1. Prepare your enviroment with the mlir-aie tools (build during Prerequisites part of this guide)

    ```
    cd <yourPathToDesignsWithMLIR-AIE>
    source <yourPathToBuildMLIR-AIE>/sandbox/bin/activate
    source yourVitisSetupScript (example shown above)
    source <yourPathToBuildMLIR-AIE>/utils/env_setup.sh <yourPathToBuildMLIR-AIE>/install <yourPathToBuildMLIR-AIE>/llvm/install
    ```
1. Goto the design of interest and run `make`

### Build and run host part: powershell

Note that your design of interest might need an adapted CMakelists.txt file. Also pay attention to accurately set the paths CMake parameters BOOST_ROOT, XRT_INC_DIR and XRT_LIB_DIR used in the CMakelists.txt, either in the file or as CMake command line parameters.

1. Build: Goto the same design of interest folder where the AIE design just got build (see above)
    ```
    mkdir buildMSVS
    cd buildMSVS
    cmake .. -G "Visual Studio 17 2022"
    cmake --build . --config Release
    ```
    
1. Run (program arguments are just an example for add_one design)
   ```
    cd Release
    .\<testName>.exe -x ..\..\build\final.xclbin -k MLIR_AIE -i ..\..\build\insts.txt -v 1
    ```
