#!/usr/bin/env bash
##===- utils/clone-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script checks out LLVM.  We use this instead of a git submodule to avoid
# excessive copies of the LLVM tree.
#
##===----------------------------------------------------------------------===##

export commithash=63f0c00d38ee7879239975a6743d4e6c7847b725 

git clone --depth 1 https://github.com/Xilinx/cmakeModules cmakeModules/cmakeModulesXilinx
export CMAKE_MODULE_PATH=`pwd`/cmakeModules/cmakeModulesXilinx

git clone --depth 1 https://github.com/llvm/llvm-project.git llvm
pushd llvm
git fetch --depth=1 origin $commithash
git checkout $commithash
popd

