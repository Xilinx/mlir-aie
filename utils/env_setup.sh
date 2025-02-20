#!/bin/bash
##===- utils/env_setup.sh - Setup mlir-aie env post build to compile IRON designs --*- Script -*-===##
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
#
# source env_setup.sh <mlir-aie install dir> 
#                     <llvm-aie/peano install dir>
#
# e.g. source env_setup.sh /scratch/mlir-aie/install 
#                          /scratch/llvm-aie/install
#
##===----------------------------------------------------------------------===##

if [ "$#" -ge 1 ]; then
    export MLIR_AIE_INSTALL_DIR=`realpath $1`
fi

if [ "$#" -ge 2 ]; then
    export PEANO_INSTALL_DIR=`realpath $2`
fi

if [[ $MLIR_AIE_INSTALL_DIR == "" ]]; then
  python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/ 
  export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
fi

if [[ $PEANO_INSTALL_DIR == "" ]]; then
  python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  export PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
fi

export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
