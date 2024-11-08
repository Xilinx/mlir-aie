#!/bin/bash
##===- utils/env_setup.sh - Setup mlir-aie env post build to compile mlir-aie designs --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script sets up the environment to use mlir-aie tools.
# The script will also download and set up llvm-aie (peano).
# 
#
# source env_setup.sh <mlir-aie install dir> 
#                     <llvm install dir> 
#                     <llvm-aie/peano install dir>
#
# e.g. source env_setup.sh /scratch/mlir-aie/install 
#                          /scratch/llvm/install
#                          /scratch/llvm-aie/install
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs 2 arguments for <mlir-aie install dir> and <llvm install dir>"
    return 1
fi

export MLIR_AIE_INSTALL_DIR=`realpath $1`
export LLVM_INSTALL_DIR=`realpath $2`
if [ "$#" -eq 3 ]; then
    export PEANO_INSTALL_DIR=`realpath $3`
fi

if [[ $PEANO_INSTALL_DIR == "" ]]; then
  mkdir -p my_install
  pushd my_install
  pip -q download llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  unzip -q llvm_aie*.whl
  rm -rf llvm_aie*.whl
  export PEANO_INSTALL_DIR=`realpath llvm-aie`
  popd
fi

export PATH=${PEANO_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH} 
export LD_LIBRARY_PATH=${PEANO_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

source ironenv/bin/activate
# Instead of adding the install python path to the $PYTHONPATH variable, add it into the
# site packages of the venv. This ensure it is picked up by jupyter servers using the venv
# as an ipykernel (and PYTHONPATH is not always percolated in these cases)
venv_site_packages=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
echo ${MLIR_AIE_INSTALL_DIR}/python > $venv_site_packages/mlir-aie.pth
#export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}