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
  XBUTIL=`which xbutil`
  if ! test -f "$XBUTIL"; then 
    echo "XRT is not installed"
    return 1
  fi
  NPU=`/opt/xilinx/xrt/bin/xbutil examine | grep RyzenAI`
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
if ! hash python3.8; then
  echo "This script requires python3.8"
  echo "https://linuxgenie.net/how-to-install-python-3-8-on-ubuntu-22-04/"
  echo "Don't forget python3.8-distutils!"
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
alias python3=python3.8
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
VPP=`which v++`
if test -f "$VPP"; then
  AIETOOLS="`dirname $VPP`/../aietools"
  mkdir -p my_install
  pushd my_install
  pip download mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/
  unzip -q mlir_aie-*_x86_64.whl
  pip download mlir -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro/
  unzip -q mlir-*_x86_64.whl
  pip install https://github.com/makslevental/mlir-python-extras/archive/d84f05582adb2eed07145dabce1e03e13d0e29a6.zip
  rm -rf mlir*.whl
  export PATH=`realpath mlir_aie/bin`:`realpath mlir/bin`:$PATH
  export LD_LIBRARY_PATH=`realpath mlir_aie/lib`:`realpath mlir/lib`:$LD_LIBRARY_PATH
  export PYTHONPATH=`realpath mlir_aie/python`:$PYTHONPATH
  popd
  python3 -m pip install --upgrade --force-reinstall -r python/requirements.txt
  pushd reference_designs/ipu-xrt
else
  echo "Vitis not found! Exiting..."
fi
