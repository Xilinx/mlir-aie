##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

# ---

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt

# ---

XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

XRT_FLAGS=-I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}
XRT_LIBS=-lxrt_coreutil
CXX=g++-13 -ggdb 

mlir_target?=build/aie.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.txt
host_target?=build/test

experimental_mlir_target?=build/experimental_aie.mlir
experimental_xclbin_target?=build/experimental_final.xclbin
experimental_insts_target?=build/experimental_insts.txt
experimental_host_target?=build/experimental_test

M=768
N=2304
m=96
n=32

.PHONY: all
all: ${xclbin_target} ${host_target}

build/kernel.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}

${mlir_target}: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< $M $N $m $n > $@

${experimental_mlir_target}: ${srcdir}/experimental_aie2.py
	mkdir -p ${@D}
	python3 $< $M $N $m $n > $@

${xclbin_target}: ${mlir_target} build/kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=${insts_target:build/%=%} ${<:%=../%}

${experimental_xclbin_target}: ${experimental_mlir_target} build/kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=${experimental_insts_target:build/%=%} ${<:%=../%}

${host_target}: ${srcdir}/test.cpp ${xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} -DM=$M -DN=$N -o $@ $< ${XRT_LIBS}

${experimental_host_target}: ${srcdir}/test.cpp ${experimental_xclbin_target}
	mkdir -p ${@D}
	mv ${experimental_xclbin_target} ${xclbin_target}
	mv ${experimental_insts_target} ${insts_target}
	${CXX} ${XRT_FLAGS} -DM=$M -DN=$N -o $@ $< ${XRT_LIBS}

.PHONY: run
run: ${host_target}
	./${host_target}

.PHONY: run_experimental
run_experimental: ${experimental_host_target}
	./${experimental_host_target}

xclbin_sign=${XILINX_XRT}/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: clean
clean:
	-rm -r build
