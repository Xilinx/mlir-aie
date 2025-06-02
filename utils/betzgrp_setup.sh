#!/bin/bash
##===- utils/env_setup.sh - Setup env for source-built mlir-aie + peano --===##
# 
# Author: James Yen
# Date: June 2025
#
# Usage:
#   source utils/env_setup.sh [mlir-aie-install-dir] [llvm-aie-wheel-dir]
#
# Optional:
#   [mlir-aie-install-dir] =: mlir-aie/install dir name, default is 'install'
#   [llvm-aie-wheel-dir]   =: llvm-aie wheel dir name, default is 
#                             'ironenv/lib/python3.13/site-packages/llvm-aie'
#
# Example:
#   source betzgrp_setup.sh 
##===----------------------------------------------------------------------===##  

# Default: mlir-aie install at ./install
BASE_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
export MLIR_AIE_INSTALL_DIR=$(realpath "${1:-$BASE_DIR/install}")

# Default: Peano from wheel under virtualenv
export PEANO_INSTALL_DIR=$(realpath "${2:-$BASE_DIR/ironenv/lib/python3.13/site-packages/llvm-aie}")

# Export env vars
export PATH="${MLIR_AIE_INSTALL_DIR}/bin:${PATH}"
export PYTHONPATH="${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
export PEANO_CLANG="${PEANO_INSTALL_DIR}/bin/clang++"

# Setup XRT if needed
if ! command -v xrt-smi &>/dev/null; then
  source /opt/xilinx/xrt/setup.sh
fi

# Set NPU2 if targeting supported NPU
NPU_INFO=$(/opt/xilinx/xrt/bin/xrt-smi examine | grep -E "NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]")
if echo "$NPU_INFO" | grep -qiE "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]"; then
  export NPU2=1
else
  export NPU2=0
fi

# Confirm
echo ""
echo "=== Environment Setup Complete ==="
echo "MLIR_AIE_INSTALL_DIR : $MLIR_AIE_INSTALL_DIR"
echo "PEANO_INSTALL_DIR     : $PEANO_INSTALL_DIR"
echo "PATH                  : $PATH"
echo "PYTHONPATH            : $PYTHONPATH"
echo "LD_LIBRARY_PATH       : $LD_LIBRARY_PATH"
echo "PEANO_CLANG           : $PEANO_CLANG"