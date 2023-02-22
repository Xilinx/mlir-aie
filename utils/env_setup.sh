#!/bin/bash
##===- utils/env_setup.sh - Setup mlir-aie env post build to compile mlir-aie designs --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script sets up the environment to run the mlir-aie build tools.
#
# source env_setup.sh <mlir-aie install dir> <llvm install dir>
#
# e.g. source env_setup.sh /scratch/mlir-aie/install /scratch/llvm/install
#
##===----------------------------------------------------------------------===##

if [ "$#" -ne 2 ]; then
    echo "ERROR: Needs 2 arguments for <mlir-aie install dir> and <llvm install dir>"
    return 1
fi

export MLIR_AIE_INSTALL_DIR=`realpath $1`
export LLVM_INSTALL_DIR=`realpath $2`

export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH} 
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

