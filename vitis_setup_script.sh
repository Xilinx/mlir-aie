#!/bin/bash
#################################################################################
# Setup Vitis (which includes aiecompiler and xchessde)
#################################################################################
export MYXILINX_VER=2024.2
export MYXILINX_BASE=/proj/xbuilds/${MYXILINX_VER}_INT_daily_latest
export XILINX_LOC=$MYXILINX_BASE/installs/lin64/Vitis/$MYXILINX_VER
export AIETOOLS_ROOT=$XILINX_LOC/aietools
export PATH=$PATH:${AIETOOLS_ROOT}/bin:$XILINX_LOC/bin
export LM_LICENSE_FILE=2100@aiengine

source ironenv/bin/activate
export NPU2=1
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh