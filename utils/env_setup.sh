#!/bin/bash
##===- utils/env_setup.sh - Setup mlir-aie env to compile IRON designs --*- Script -*-===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script sets up the environment to compile IRON designs.
# The script will download and set up mlir-aie and llvm-aie (peano).
#
# source env_setup.sh [--force-install] <mlir-aie install dir>
#                                      <llvm-aie/peano install dir>
#
# e.g. source env_setup.sh /scratch/mlir-aie/install /scratch/llvm-aie/install
#
##===----------------------------------------------------------------------===##

FORCE_INSTALL=0
if [ "$1" = "--force-install" ]; then
  FORCE_INSTALL=1
  shift
fi

if [ "$#" -ge 1 ]; then
    export MLIR_AIE_INSTALL_DIR=`realpath $1`
    export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH}
    export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
    export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
    FORCE_INSTALL=0
else
    export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie 2>/dev/null | grep ^Location: | awk '{print $2}')/mlir_aie"
fi

if [ "$#" -ge 2 ]; then
    export PEANO_INSTALL_DIR=`realpath $2`
    FORCE_INSTALL=0
else
    export PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | grep ^Location: | awk '{print $2}')/llvm-aie"
fi

# If force install or an install dir isn't passed
if [[ $FORCE_INSTALL -eq 1 || ( "$#" -lt 1 && -z "$(pip show mlir_aie | grep ^Location:)" ) ]]; then
  python3 -m pip install -I mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2
  export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
fi

# If force install or an install dir isn't passed
if [[ $FORCE_INSTALL -eq 1 || ( "$#" -lt 2 && -z "$(pip show llvm-aie | grep ^Location:)" ) ]]; then
  python3 -m pip install -I llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  export PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
fi

XRTSMI=`which xrt-smi`
if ! test -f "$XRTSMI"; then
  source /opt/xilinx/xrt/setup.sh
fi
NPU=`/opt/xilinx/xrt/bin/xrt-smi examine | grep -E "NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]"`
NPU="${NPU:-$(/mnt/c/Windows/System32/AMD/xrt-smi.exe examine 2>/dev/null | tr -d '\r' | grep -E 'NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]' || true)}"
# Check if the current environment is NPU2
# npu4 => Strix, npu5 => Strix Halo, npu6 => Krackan
if echo "$NPU" | grep -qiE "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]"; then
    export NPU2=1
else
    export NPU2=0
fi

echo ""
echo "Note: Peano (llvm-aie) has not been added to PATH to avoid conflict with"
echo "      system clang/clang++. It can be found in: \$PEANO_INSTALL_DIR/bin"
echo ""
echo "PATH              : $PATH"
echo "LD_LIBRARY_PATH   : $LD_LIBRARY_PATH"
echo "PYTHONPATH        : $PYTHONPATH"
