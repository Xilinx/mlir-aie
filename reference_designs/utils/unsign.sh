#!/bin/bash
source /opt/xilinx/xrt/setup.sh
#/opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh -dev Phoenix -xclbin $1
/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh -dev Phoenix -xclbin $1
