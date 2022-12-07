#!/bin/bash
##===- utils/single_unit_test_compile.sh - Wrapper to compile single unit test --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script simplifies the aiecc.py compile command for compiling the unit
# tests given the sysyroot dir and runtime lib dir. The default is compiling
# for libxaie v2 drivers.
#
# single_unit_test_compile.sh <sysroot dir> <runtime lib dir>
#
# e.g. single_unit_test_compile.sh /scratch/vck190_bare_prod_sysroot 
#      /scratch/mlir-aie/runtime_lib
#
##===----------------------------------------------------------------------===##

if [ "$#" -ne 2 ]; then
    echo "ERROR: Needs 2 arguments for <sysroot dir> and <runtime lib dir>"
    exit 1
fi

export SYSROOT_DIR=$1
export RUNTIME_LIB_DIR=$2

#aiecc.py --sysroot=$SYSROOT_DIR aie.mlir -I${RUNTIME_LIB_DIR} \
# ${RUNTIME_LIB_DIR}/test_library.cpp ./test.cpp -o test.elf

aiecc.py --sysroot=$SYSROOT_DIR --aie-generate-xaiev2 ./aie.mlir \
    -DLIBXAIENGINEV2 -I${RUNTIME_LIB_DIR} ${RUNTIME_LIB_DIR}/test_library.cpp \
    ./test.cpp -o test.elf
    