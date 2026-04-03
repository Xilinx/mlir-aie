##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=single_core_dynamic
targetname=single_core_dynamic

# Dynamic design always uses bf16->f32, m=k=n=32, npu2 only
m=32
k=32
n=32
M?=128
K?=128
N?=128
dtype_in=bf16
dtype_out=f32
devicename=npu2

kernels=mm_${m}x${k}x${n}

use_placed?=0
use_iron?=0

ifeq (${use_placed}, 1)
aie_py_src=single_core_dynamic_placed.py
else ifeq (${use_iron}, 1)
aie_py_src=single_core_dynamic_iron.py
else
aie_py_src=single_core_dynamic.py
endif

KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n}

include ${srcdir}/../makefile-common

build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

# Dynamic test executable (C++ TXN-based runner)
MLIR_AIE_SRC_DIR ?= $(shell cd ${srcdir}/../../../.. && pwd)
XRT_INC_DIR ?= /opt/xilinx/xrt/include
XRT_LIB_DIR ?= /opt/xilinx/xrt/lib
TEST_UTILS_DIR ?= ${MLIR_AIE_SRC_DIR}/runtime_lib/test_lib

dynamic_single_core.exe: ${srcdir}/test_dynamic.cpp ${srcdir}/dynamic_gemm_txn.h
	${CXX} -std=c++23 -O2 -ggdb -o $@ $< \
		-I${MLIR_AIE_SRC_DIR}/include \
		-I${srcdir} -I${srcdir}/.. \
		-I${XRT_INC_DIR} -I${TEST_UTILS_DIR} \
		-I${MLIR_AIE_SRC_DIR}/runtime_lib \
		-L${XRT_LIB_DIR} -lxrt_coreutil

.PHONY: run_dynamic
run_dynamic: dynamic_single_core.exe ${xclbin_target}
	./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M 32 -K 32 -N 32 -v 1
	./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M 64 -K 64 -N 64 -v 1
	./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M ${M} -K ${K} -N ${N} -v 1

# --- Auto-generated TXN targets (via aiecc --aie-generate-txn-cpp) ---

# Generate dynamic MLIR with SSA M/K/N parameters in runtime_sequence
build/dynsize/aie_gemm_dynamic.mlir: ${srcdir}/single_core_dynamic.py
	mkdir -p ${@D}
	python3 $< --dynamic-txn -M ${M} -K ${K} -N ${N} > $@

# Build XCLBIN AND generate C++ TXN in one unified aiecc invocation.
# RuntimeSequenceOp has IsolatedFromAbove trait, so SCF→CF skips it
# automatically — preserving SCF ops for EmitC while lowering core bodies.
# Both XCLBIN and TXN use identical buffer addresses from the same MLIR.
build/dynsize/final_dynamic.xclbin build/dynsize/generated_gemm_txn.h &: build/dynsize/aie_gemm_dynamic.mlir build/mm_${m}x${k}x${n}.o
	mkdir -p build/dynsize
	cp build/mm_${m}x${k}x${n}.o build/dynsize/
	cd build/dynsize && aiecc --alloc-scheme=basic-sequential \
		--aie-generate-xclbin --xclbin-name=final_dynamic.xclbin \
		--aie-generate-txn-cpp --txn-cpp-name=generated_gemm_txn.h \
		--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
		$(<:%=../../%)

# Build test executable using auto-generated TXN (no --instr needed)
dynamic_generated.exe: ${srcdir}/test_dynamic.cpp build/dynsize/generated_gemm_txn.h
	${CXX} -std=c++23 -O2 -ggdb -o $@ $< \
		-DUSE_GENERATED_TXN \
		-Ibuild/dynsize \
		-I${MLIR_AIE_SRC_DIR}/include \
		-I${srcdir} -I${srcdir}/.. \
		-I${XRT_INC_DIR} -I${TEST_UTILS_DIR} \
		-I${MLIR_AIE_SRC_DIR}/runtime_lib \
		-L${XRT_LIB_DIR} -lxrt_coreutil

.PHONY: run_dynamic_generated
run_dynamic_generated: dynamic_generated.exe build/dynsize/final_dynamic.xclbin
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 32 -K 32 -N 32 -v 1
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 64 -K 64 -N 64 -v 1
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 96 -K 96 -N 96 -v 1
	./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE -M 128 -K 128 -N 128 -v 1
