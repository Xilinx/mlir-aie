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
# single_unit_test_compile.sh <runTimeArch> <sysroot dir>
#
# e.g. single_unit_test_compile.sh aarch64 /scratch/vck190_bare_prod_sysroot
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 1 ]; then
    echo "ERROR: Need 1 argument <runtime architecture {aarch64,x86_64} and optionaly <sysroot dir>."
    exit 1
fi

runTimeArch=$1

VITIS_ROOT=`realpath $(dirname $(which vitis))/../`

SYSROOT_DIR=${2:-"$VITIS_ROOT/gnu/aarch64/lin/aarch64-linux/aarch64-xilinx-linux"}
extraAieCCFlags=""

if [ "$runTimeArch" == "aarch64" ] && [ "$#" -eq 1 ]; then
    LIBCXX_VERSION="11.2.0"
    extraAieCCFlags+="-I$SYSROOT_DIR/usr/include/c++/$LIBCXX_VERSION -I$SYSROOT_DIR/usr/include/c++/$LIBCXX_VERSION/aarch64-xilinx-linux -L$SYSROOT_DIR/usr/lib/aarch64-xilinx-linux/$LIBCXX_VERSION -B$SYSROOT_DIR/usr/lib/aarch64-xilinx-linux/$LIBCXX_VERSION"
fi

runtimeLibs=`realpath $(dirname $(which aie-opt))/../runtime_lib`

if [ "$runTimeArch" == "aarch64" ]; then
    runtimeLibs+=/aarch64
    
    aiecc.py -v --sysroot=$SYSROOT_DIR \
        --host-target=aarch64-linux-gnu ./aie.mlir \
        -I${runtimeLibs}/test_lib/include -L${runtimeLibs}/test_lib/lib -ltest_lib $extraAieCCFlags \
        ./test.cpp -o test.elf
elif [ "$runTimeArch" == "x86_64" ]; then
    runtimeLibs+=/x86_64
    
	  aiecc.py -v --host-target=x86_64-amd-linux-gnu  aie.mlir \
						-I${runtimeLibs}/test_lib/include ${runtimeLibs}/test_lib/src/test_library.cpp \
						test.cpp -Wl,--no-whole-archive -lstdc++ -ldl -o test.elf
elif [ "$runTimeArch" == "x86_64-hsa" ]; then
    runtimeLibs+=/x86_64-hsa
    
	  aiecc.py -v --host-target=x86_64-amd-linux-gnu  --link_against_hsa aie.mlir \
						-I${runtimeLibs}/test_lib/include ${runtimeLibs}/test_lib/src/test_library.cpp \
						test.cpp -Wl,--no-whole-archive -lstdc++ -ldl -o test.elf

else
    echo "Error: unsupported runtime architecture"
fi
