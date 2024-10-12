#!/usr/bin/env bash
##===- setup_iron_env.sh - Setup IRON for Ryzen AI dev -------*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script is used during and after the installing IRON using the 
# quick_setup.sh script to add the tools to your environment.
#
# source ./utils/setup_iron_env.sh
#
##===----------------------------------------------------------------------===##

if [[ $WSL_DISTRO_NAME == "" ]]; then
  XRTSMI=`which xrt-smi`
  if ! test -f "$XRTSMI"; then 
    echo "XRT is not installed"
    return 1
  fi
  NPU=`/opt/xilinx/xrt/bin/xrt-smi examine | grep RyzenAI`
  if [[ $NPU == *"RyzenAI"* ]]; then
    echo "Ryzen AI NPU found:"
    echo $NPU
  else
    echo "NPU not found. Is the amdxdna driver installed?"
    return 1
  fi
else
  echo "Environment is WSL"
fi

pushd my_install
export PATH=`realpath llvm-aie/bin`:`realpath mlir_aie/bin`:`realpath mlir/bin`:$PATH
export LD_LIBRARY_PATH=`realpath llvm-aie/lib`:`realpath mlir_aie/lib`:`realpath mlir/lib`:$LD_LIBRARY_PATH
export PYTHONPATH=`realpath mlir_aie/python`:$PYTHONPATH
export PEANO_DIR=`realpath llvm-aie`
popd

AIEOPT=`which aie-opt`
if ! test -f "$AIEOPT"; then 
  echo "IRON tools are not found, please reinstall using quick_setup.sh"
  return 1
else 
  echo "IRON environment setup complete"
fi
