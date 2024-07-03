##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

# ---

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt
XILINX_VITIS?=/tools/Xilinx/Vitis/2023.2

# ---

XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

CXX=g++-13 -ggdb -I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}

CHESSCCWRAP2_FLAGS = aie2 -I${XILINX_VITIS}/aietools/include 

mlir_target?=build/aie.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.txt
host_target?=build/test

M=768
N=2304
m=96
n=32

.PHONY: all
all: ${xclbin_target} ${host_target}

build/kernel.o: kernel.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DDIM_m=$m -DDIM_n=$n -c ${<:%=../%} -o ${@F}

${mlir_target}: aie2.py
	mkdir -p ${@D}
	python3 $< $M $N $m $n > $@

${xclbin_target}: ${mlir_target} build/kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${host_target}: test.cpp ${xclbin_target}
	mkdir -p ${@D}
	${CXX} -DM=$M -DN=$N -o $@ $< -lxrt_coreutil

.PHONY: run
run: ${host_target}
	./${host_target}

xclbin_sign=${XILINX_XRT}/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: clean
clean:
	-rm -r build
