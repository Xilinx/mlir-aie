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

# Set SRC_DIR to the directory in which this Makefile sits.
SRC_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# We need to find the root directories for the following dependencies:
# - Xilinx XRT (runtime library) for the host code
# - MLIR-AIE with AIE-API headers for the kernel code
XILINX_XRT_DIR?=$(shell realpath $(dir $(shell which xrt-smi))/..)
MLIR_AIE_OPT_DIR?=$(shell realpath $(dir $(shell which aie-opt))/..)

# Configuration of this example.
DEVICE_NAME?=$(if $(filter 1,$(NPU2)),npu2,npu)

# Host C++ compiler selection / flags
HOST_CXX?=g++-13
HOST_CFLAGS?=-I${XILINX_XRT_DIR}/include -L${XILINX_XRT_DIR}/lib
HOST_LDLIBS?=-lxrt_coreutil

# Kernel C++ compiler selection / flags
KERNEL_CXX?=${PEANO_INSTALL_DIR}/bin/clang++
KERNEL_CFLAGS?=-O2 -std=c++20 -I${MLIR_AIE_OPT_DIR}/include
ifeq (${DEVICE_NAME}, npu)
	KERNEL_CFLAGS+=--target=aie2-none-unknown-elf
endif
ifeq (${DEVICE_NAME}, npu2)
	KERNEL_CFLAGS+=--target=aie2p-none-unknown-elf
endif


# ----------------------------------------------------------------------------
# Build Targets
# ----------------------------------------------------------------------------

.PHONY: all clean run
all: build/final.xclbin build/insts.bin build/test

# Create the build directory
build:
	mkdir -p $@

# Host code: Program executed on the CPU that will invoke our NPU design
# generated below and verify its outputs.
build/test: ${SRC_DIR}/test.cpp | build
	${HOST_CXX} ${HOST_CFLAGS} -o $@ $< ${HOST_LDLIBS}

# The final.xclbin contains static configuration for the NPU (including data 
# movement configuration, compure core program memories). 
build/final.xclbin: build/aie.mlir build/matrix_multiplication.o | build
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --xclbin-name=${@F} --no-xchesscc --no-xbridge ${<:%=../%}

build/insts.bin: build/aie.mlir | build
# insts.bin contains instructions for the command processor to execute
# at runtime (i.e., dispatch running the design, read from DRAM, etc.).
	cd ${@D} && aiecc.py -v --aie-generate-npu-insts --npu-insts-name=${@F} --no-xchesscc --no-xbridge ${<:%=../%}

# Lower-level MLIR design built from higher-level Python IRON design
build/aie.mlir: ${SRC_DIR}/matrix_multiplication_single_core.py | build
	python3 $< ${DEVICE_NAME} > $@

# External single compute core kernel code defined in C source
build/matrix_multiplication.o: ${SRC_DIR}/matrix_multiplication.cc | build
	cd ${@D} && ${KERNEL_CXX} ${KERNEL_CFLAGS} -c $< -o ${@F}

run: build/test build/final.xclbin build/insts.bin
	./build/test build/final.xclbin build/insts.bin MLIR_AIE

clean: 
	rm -rf build
