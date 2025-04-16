#!/bin/bash

#################################################################################
# Setup Vitis AIE Essentials
#################################################################################
export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
export PATH=$PATH:${AIETOOLS_ROOT}/bin
export LM_LICENSE_FILE=/opt/Xilinx.lic

source ironenv/bin/activate
export NPU2=1
source /opt/xilinx/xrt/setup.sh
source utils/env_setup.sh