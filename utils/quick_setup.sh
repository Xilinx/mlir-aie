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
if hash python3.12; then
   echo "Using python version `python3.12 --version`"
   my_python=python3.12
elif hash python3.10; then
   echo "Using python version `python3.10 --version`"
   my_python=python3.10
else
   echo "This script requires python3.10 or python3.12"
   return 1
fi
if ! hash unzip; then
  echo "unzip is not installed"
  return 1
fi
# if an install is already present, remove it to start from a clean slate
rm -rf ironenv
rm -rf my_install
$my_python -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip
VPP=`which xchesscc`
if test -f "$VPP"; then
  mkdir -p my_install
  pushd my_install
  pip download mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/
  unzip -q mlir_aie-*_x86_64.whl
  pushd mlir_aie/python
  pip download aie_python_bindings -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/
  unzip -q -o aie_python_bindings*.whl
  rm *.whl
  popd
  pip download mlir -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro/
  unzip -q mlir-*_x86_64.whl
  pip -q download llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  unzip -q llvm_aie*.whl
  rm -rf mlir*.whl
  rm -rf llvm_aie*.whl
  export PEANO_INSTALL_DIR=`realpath llvm-aie`
  popd
  python3 -m pip install -r python/requirements.txt
  HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
  python3 -m pip install -r python/requirements_ml.txt
  source utils/env_setup.sh my_install/mlir_aie my_install/mlir
  pushd programming_examples
  echo "PATH              : $PATH"
  echo "LD_LIBRARY_PATH   : $LD_LIBRARY_PATH"
  echo "PYTHONPATH        : $PYTHONPATH"
else
  echo "Vitis not found! Exiting..."
fi
