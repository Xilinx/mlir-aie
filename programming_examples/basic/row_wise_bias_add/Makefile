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

XILINX_XRT=/opt/xilinx/xrt/
XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

XRT_FLAGS=-I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}
XRT_LIBS=-lxrt_coreutil
CXX=g++-13 -ggdb

UTILS_INCLUDE := -I$(srcdir)/../../../runtime_lib/test_lib/
UTILS_LIB=$(srcdir)/../../../runtime_lib/test_lib/test_utils.cpp


mlir_target?=build/aie.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.bin
host_target?=build/test

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
M=768
N=2304
m=96
n=32

aie_py_src=row_wise_bias_add.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=row_wise_bias_add_placed.py
endif

.PHONY: all
all: ${xclbin_target} ${host_target}

build/kernel.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}
endif

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} $M $N $m $n > $@

${xclbin_target}: ${mlir_target} build/kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=${insts_target:build/%=%} ${<:%=../%}

${host_target}: ${srcdir}/test.cpp ${xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB} -DSIZE_M=$M -DSIZE_N=$N -o $@ $< ${XRT_LIBS}

.PHONY: run
run: ${host_target}
	./${host_target}

.PHONY: clean
clean:
	-rm -r build