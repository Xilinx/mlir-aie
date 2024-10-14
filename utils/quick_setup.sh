#!/usr/bin/env bash
##===- quick_setup.sh - Setup IRON for Ryzen AI dev ----------*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script is the quickest path to running the Ryzen AI reference designs.
# Please have the Vitis tools and XRT environment setup before sourcing the 
# script.
#
# source ./utils/quick_setup.sh
#
##===----------------------------------------------------------------------===##

echo "Setting up RyzenAI developement tools..."
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
if ! hash python3.10; then
   echo "This script requires python3.10"
   return 1
fi
if ! hash virtualenv; then
  echo "virtualenv is not installed"
  return 1
fi
if ! hash unzip; then
  echo "unzip is not installed"
  return 1
fi
# if an install is already present, remove it to start from a clean slate
rm -rf ironenv
rm -rf my_install
python3 -m virtualenv ironenv
# The real path to source might depend on the virtualenv version
if [ -r ironenv/local/bin/activate ]; then
  source ironenv/local/bin/activate
else
  source ironenv/bin/activate
fi
python3 -m pip install --upgrade pip
VPP=`which xchesscc`
if test -f "$VPP"; then
  mkdir -p my_install
  pushd my_install
  pip download mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/
  unzip -q mlir_aie-*_x86_64.whl
  pip download mlir -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro/
  unzip -q mlir-*_x86_64.whl
  pip -q download llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  unzip -q llvm_aie*.whl
  rm -rf mlir*.whl
  rm -rf llvm_aie*.whl
  export PATH=`realpath llvm-aie/bin`:`realpath mlir_aie/bin`:`realpath mlir/bin`:$PATH
  export LD_LIBRARY_PATH=`realpath llvm-aie/lib`:`realpath mlir_aie/lib`:`realpath mlir/lib`:$LD_LIBRARY_PATH
  export PYTHONPATH=`realpath mlir_aie/python`:$PYTHONPATH
  export PEANO_DIR=`realpath llvm-aie`
  popd
  python3 -m pip install --upgrade --force-reinstall --no-cache-dir -r python/requirements.txt
  HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install --upgrade --force-reinstall --no-cache-dir -r python/requirements_extras.txt
  python3 -m pip install --upgrade --force-reinstall --no-cache-dir -r python/requirements_ml.txt
  pushd programming_examples
else
  echo "Vitis not found! Exiting..."
fi
