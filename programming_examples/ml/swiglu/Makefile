##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SRC_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# We need to find the root directories for the following dependencies:
# - Xilinx XRT (runtime library) for the host code
# - MLIR-AIE with AIE-API headers for the kernel code
XILINX_XRT_DIR?=$(shell realpath $(dir $(shell which xrt-smi))/..)
MLIR_AIE_OPT_DIR?=$(shell realpath $(dir $(shell which aie-opt))/..)
RUNTIME_LIB_DIR?=${MLIR_AIE_OPT_DIR}/runtime_lib/x86_64

# Configuration of this example.
DEVICE_NAME?=$(if $(filter 1,$(NPU2)),npu2,npu)

# Host C++ compiler selection / flags
HOST_CXX?=g++-13
HOST_CFLAGS?=-std=c++23 -I${XILINX_XRT_DIR}/include -L${XILINX_XRT_DIR}/lib -I${RUNTIME_LIB_DIR}/test_lib/include -L${RUNTIME_LIB_DIR}/test_lib/lib
HOST_LDLIBS?=-lxrt_coreutil -ltest_utils

# Kernel C++ compiler selection / flags
KERNEL_CXX?=${PEANO_INSTALL_DIR}/bin/clang++
KERNEL_CFLAGS?=-O2 -std=c++20 -I${MLIR_AIE_OPT_DIR}/include -DNDEBUG \
	-Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
	-Wno-missing-template-arg-list-after-template-kw 
ifeq (${DEVICE_NAME}, npu)
	KERNEL_CFLAGS+=--target=aie2-none-unknown-elf
endif
ifeq (${DEVICE_NAME}, npu2)
	KERNEL_CFLAGS+=--target=aie2p-none-unknown-elf
endif
AIE2_KERNELS_DIR=${SRC_DIR}/../../../aie_kernels/aie2
AIE2P_KERNELS_DIR=${SRC_DIR}/../../../aie_kernels/aie2p


# ----------------------------------------------------------------------------
# Build Targets
# ----------------------------------------------------------------------------

.PHONY: all clean run
all: build/swiglu build/combined.xclbin

build:
	mkdir -p $@

# Host code
# --------------------------------------------------------------------------

build/swiglu: ${SRC_DIR}/swiglu.cpp | build
	${HOST_CXX} ${HOST_CFLAGS} -o $@ $< ${HOST_LDLIBS}

# AIE code
# --------------------------------------------------------------------------

seq=2048
dim=2048
seq_x_dim=4194304

build/matmul.mlir: ${SRC_DIR}/../../basic/matrix_multiplication/whole_array/whole_array.py | build
	sed 's/mm_{m}x{k}x{n}.o/matmul.o/g' $< > ${@D}/matmul.py
	python3 ${@D}/matmul.py --dev ${DEVICE_NAME} -M ${dim} -K ${dim} -N ${seq} -m 64 -k 64 -n 64 --n-aie-cols 4 --dtype_in bf16 --dtype_out bf16  > $@

build/matmul.o: ${AIE2P_KERNELS_DIR}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CXX} ${KERNEL_CFLAGS} -Dbf16_bf16_ONLY -DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -c $< -o ${@F}

build/silu.mlir: ${SRC_DIR}/../silu/silu.py | build
	sed 's/kernels.a/silu.o/g' $< > ${@D}/silu.py
	python3 ${@D}/silu.py --dev ${DEVICE_NAME} -l ${seq_x_dim} -co 4 -ch 2 > $@

build/silu.o: ${AIE2P_KERNELS_DIR}/silu.cc | build
	cd ${@D} && ${KERNEL_CXX} ${KERNEL_CFLAGS} -c $< -o ${@F}

build/eltwise_mul.mlir: ${SRC_DIR}/../eltwise_mul/eltwise_mul.py | build
	sed 's/mul.o/eltwise_mul.o/g' $< > ${@D}/eltwise_mul_tmp.py
	sed 's/my_eltwise_mul(dev, 65536, trace_size)/my_eltwise_mul(dev, ${seq_x_dim}, trace_size)/g' ${@D}/eltwise_mul_tmp.py > ${@D}/eltwise_mul.py
	python3 ${@D}/eltwise_mul.py ${DEVICE_NAME} ${seq} > $@

build/eltwise_mul.o: ${AIE2_KERNELS_DIR}/mul.cc | build
	cd ${@D} && ${KERNEL_CXX} ${KERNEL_CFLAGS} -c $< -o ${@F}

DESIGN_0=matmul
DESIGN_1=silu
DESIGN_2=eltwise_mul
build/combined.xclbin build/${DESIGN_0}.bin build/${DESIGN_1}.bin build/${DESIGN_2}.bin: \
		build/${DESIGN_0}.mlir build/${DESIGN_1}.mlir build/${DESIGN_2}.mlir \
		build/${DESIGN_0}.o build/${DESIGN_1}.o build/${DESIGN_2}.o \
		| build
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --no-aiesim --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-xclbin --xclbin-name=${DESIGN_0}.xclbin  \
		--aie-generate-npu-insts --npu-insts-name=${DESIGN_0}.bin \
		--xclbin-kernel-name=${DESIGN_0} --xclbin-kernel-id=0x901 --xclbin-instance-name=${DESIGN_0}_inst \
		${DESIGN_0}.mlir
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --no-aiesim --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-xclbin --xclbin-name=${DESIGN_1}.xclbin  \
		--aie-generate-npu-insts --npu-insts-name=${DESIGN_1}.bin \
		--xclbin-kernel-name=${DESIGN_1} --xclbin-kernel-id=0x902 --xclbin-instance-name=${DESIGN_1}_inst \
		--xclbin-input=${DESIGN_0}.xclbin \
		${DESIGN_1}.mlir
	cd ${@D} && aiecc.py -v --no-xchesscc --no-xbridge --no-aiesim --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-xclbin --xclbin-name=${DESIGN_2}.xclbin  \
		--aie-generate-npu-insts --npu-insts-name=${DESIGN_2}.bin \
		--xclbin-kernel-name=${DESIGN_2} --xclbin-kernel-id=0x903 --xclbin-instance-name=${DESIGN_2}_inst \
		--xclbin-input=${DESIGN_1}.xclbin \
		${DESIGN_2}.mlir
	cp build/${DESIGN_2}.xclbin $@


# Phony targets
# --------------------------------------------------------------------------

run: build/swiglu build/combined.xclbin build/matmul.bin build/silu.bin build/eltwise_mul.bin
	./build/swiglu -x build/combined.xclbin \
		--insts-matmul build/matmul.bin --insts-silu build/silu.bin --insts-eltwise-mul build/eltwise_mul.bin \
		--dim 2048 --epsilon 0.2 --abs_th 1

clean: 
	rm -rf build