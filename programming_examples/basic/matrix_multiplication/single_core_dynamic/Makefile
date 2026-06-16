##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=single_core_dynamic
targetname=single_core_dynamic

# Dynamic design always uses bf16->f32, m=k=n=32
m=32
k=32
n=32
M?=128
K?=128
N?=128
dtype_in=bf16
dtype_out=f32
devicename?=npu2

aie_py_src=single_core_dynamic_txn.py

include ${srcdir}/../makefile-common

# --- Auto-generated TXN targets (driven by single_core_dynamic_txn.py) ---

MLIR_AIE_ROOT ?= $(if $(MLIR_AIE_INSTALL_DIR),$(MLIR_AIE_INSTALL_DIR),$(shell cd ${srcdir}/../../../.. && pwd))
RUNTIME_LIB_ROOT ?= $(if $(MLIR_AIE_INSTALL_DIR),${MLIR_AIE_ROOT}/runtime_lib/x86_64,${MLIR_AIE_ROOT}/runtime_lib)
XRT_INC_DIR ?= /opt/xilinx/xrt/include
XRT_LIB_DIR ?= /opt/xilinx/xrt/lib
TEST_UTILS_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),${RUNTIME_LIB_ROOT}/test_lib/include,${RUNTIME_LIB_ROOT}/test_lib)

# Build XCLBIN and generate the C++ TXN header from the @iron.jit design.
# The driver drives compile_mlir_module, which compiles the content-hashed
# kernel .o itself and runs aiecc with --aie-generate-txn-cpp (SCF is kept legal
# in aie.runtime_sequence so it survives for the TXN-C++ path). Both XCLBIN and
# TXN come from the same MLIR with identical buffer addresses.
build/dynsize/final_dynamic.xclbin build/dynsize/generated_gemm_txn.h &: ${srcdir}/single_core_dynamic_txn.py ${srcdir}/single_core_dynamic.py
	mkdir -p build/dynsize
	python3 $< -d ${devicename} -m ${m} -k ${k} -n ${n} \
	    --dtype_in ${dtype_in} --dtype_out ${dtype_out} \
	    --max-m ${M} --max-k ${K} --max-n ${N} \
	    --xclbin-path build/dynsize/final_dynamic.xclbin

# Build test executable using auto-generated TXN (no --instr needed)
dynamic_generated.exe: ${srcdir}/test_dynamic.cpp build/dynsize/generated_gemm_txn.h
	${CXX} -std=c++23 -O2 -ggdb -o $@ $< \
		-Ibuild/dynsize \
		-I${MLIR_AIE_ROOT}/include \
		-I${srcdir} -I${srcdir}/.. \
		-I${XRT_INC_DIR} -I${TEST_UTILS_DIR} \
		-I${RUNTIME_LIB_ROOT} \
		-L${XRT_LIB_DIR} -lxrt_coreutil

.PHONY: run_dynamic_generated
run_dynamic_generated: dynamic_generated.exe build/dynsize/final_dynamic.xclbin
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 32 -K 32 -N 32 -v 1
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 64 -K 64 -N 64 -v 1
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 96 -K 96 -N 96 -v 1
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 128 -K 128 -N 128 -v 1

.PHONY: run_dynamic
run_dynamic: run_dynamic_generated
