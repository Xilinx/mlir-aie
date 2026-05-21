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
# Low-level placed IRON design (see packet_switch.py docstring).
# Build add_mul.o once into build/ via peano; both --op modes link with
# it.  Each --op produces its own xclbin + insts pair via
# compile_mlir_module's --xclbin-path / --insts-path.

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

DEVICE ?= $(if $(filter 1,$(NPU2)),npu2,npu)

aie_py_src = packet_switch.py

add_xclbin_target  = build/add.xclbin
add_insts_target   = build/add_insts.bin
mult_xclbin_target = build/mult.xclbin
mult_insts_target  = build/mult_insts.bin

host_target = build/test

.PHONY: all
all: ${add_xclbin_target} ${mult_xclbin_target} ${host_target}

build/add_mul.o: ${srcdir}/add_mul.cc
	mkdir -p ${@D}
ifeq (${DEVICE}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

${add_xclbin_target} ${add_insts_target} &: ${srcdir}/${aie_py_src} build/add_mul.o
	mkdir -p $(dir ${add_xclbin_target})
	python3 $< -d ${DEVICE} --op add --xclbin-path=${add_xclbin_target} --insts-path=${add_insts_target}

${mult_xclbin_target} ${mult_insts_target} &: ${srcdir}/${aie_py_src} build/add_mul.o
	mkdir -p $(dir ${mult_xclbin_target})
	python3 $< -d ${DEVICE} --op mul --xclbin-path=${mult_xclbin_target} --insts-path=${mult_insts_target}

${host_target}: ${srcdir}/test.cpp ${add_xclbin_target} ${mult_xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB} -o $@ $< ${XRT_LIBS}

.PHONY: run_mul run_add run
run_mul: all
	./${host_target} 1 ${mult_insts_target} ${mult_xclbin_target}

run_add: all
	./${host_target} 0 ${add_insts_target} ${add_xclbin_target}

run: run_mul run_add

.PHONY: clean
clean:
	-rm -rf build
