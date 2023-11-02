#!/usr/bin/env bash
##===- utils/build-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with the standard options. Intended to be called from 
# the github workflows.
#
##===----------------------------------------------------------------------===##

BUILD_DIR=${1:-"build"}
INSTALL_DIR=${2:-"install"}

mkdir -p llvm/$BUILD_DIR
mkdir -p llvm/$INSTALL_DIR
cd llvm/$BUILD_DIR
set -o pipefail
set -e
cmake ../llvm \
  -GNinja \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS='mlir' \
  -DLLVM_OPTIMIZED_TABLEGEN=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build . --target install -- -j$(nproc)
