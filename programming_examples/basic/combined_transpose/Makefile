##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
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


devicename?=$(if $(filter 1,$(NPU2)),npu2,npu)
M?=4096
N?=4096
m?=64
n?=32
s?=8
dtype?=i32

KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++
KERNEL_CFLAGS= 
ifeq (${devicename}, npu2)
	KERNEL_CFLAGS+=${PEANOWRAP2P_FLAGS}
else
	KERNEL_CFLAGS+=${PEANOWRAP2_FLAGS}
endif
KERNEL_DEFINES=-DDIM_m=$m -DDIM_n=$n -DDIM_s=$s -DDTYPE_${dtype}

HOST_DEFINES=-DDTYPE_${dtype}

.PHONY: all
all: build/final.xclbin build/test

build/transpose.o: ${srcdir}/transpose.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

build/aie.mlir: ${srcdir}/combined_transpose.py
	mkdir -p ${@D}
	python3 $< ${devicename} $M $N $m $n $s --dtype ${dtype} > $@

build/final.xclbin: build/aie.mlir build/transpose.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.bin ${<:%=../%}

build/test: ${srcdir}/test.cpp build/final.xclbin
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB} ${HOST_DEFINES} -o $@ $< ${XRT_LIBS}

.PHONY: run
run: build/test
	./build/test -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -M ${M} -N ${N}


.PHONY: clean
clean:
	rm -rf build
