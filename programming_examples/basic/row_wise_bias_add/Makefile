##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# NPU path uses @iron.jit's compile() directly via --xclbin-path /
# --insts-path.  @iron.jit also handles the kernel.cc → kernel.o build
# (via ExternalFunction.compile_flags carrying -DDIM_m / -DDIM_n), so
# there's no separate build/kernel.o rule.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

XILINX_XRT ?= $(if $(wildcard /opt/xilinx/xrt/include),/opt/xilinx/xrt,/usr)
XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

XRT_FLAGS=-I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}
XRT_LIBS=-lxrt_coreutil
CXX=g++-13 -ggdb

UTILS_INCLUDE := -I$(srcdir)/../../../runtime_lib/test_lib/
UTILS_LIB=$(srcdir)/../../../runtime_lib/test_lib/test_utils.cpp

xclbin_target?=build/final.xclbin
insts_target?=build/insts.bin
host_target?=build/test

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
M=768
N=2304
m=96
n=32

aie_py_src = row_wise_bias_add.py
aieargs = -d ${devicename} -M $M -N $N -m $m -n $n

.PHONY: all
all: ${xclbin_target} ${host_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
# kernel source is an additional prerequisite for the xclbin
${xclbin_target} ${insts_target}: ${srcdir}/kernel.cc

${host_target}: ${srcdir}/test.cpp ${xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB} -DSIZE_M=$M -DSIZE_N=$N -o $@ $< ${XRT_LIBS}

.PHONY: run
run: ${host_target}
	./${host_target}

.PHONY: clean
clean:
	-rm -r build
