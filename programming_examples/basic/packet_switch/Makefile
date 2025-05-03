#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.

DEVICE ?= $(if $(filter 1,$(NPU2)),npu2,npu)
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${HOME_DIR}/../../makefile-common

# Kernel
KERNEL_O_DIR := build

KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernel/*.cc)
KERNEL_OBJS := $(patsubst ${HOME_DIR}/kernel/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernel/*.h)


BITSTREAM_O_DIR := build/bitstream

XCLBIN_MLIR_SRCS += build/bitstream/aie_mul.mlir
XCLBIN_MLIR_SRCS += build/bitstream/aie_add.mlir


INSTS_TARGET := ${BITSTREAM_O_DIR}/insts_mul.txt
INSTS_TARGET += ${BITSTREAM_O_DIR}/insts_add.txt
XCLBIN_TARGET := ${BITSTREAM_O_DIR}/final.xclbin

# Host stuff
HOST_SRCDIR := ${HOME_DIR}/host
HOST_O_DIR := build/host
HOST_C_TARGET := packet_switch.exe

HOST_HEADERS = $(wildcard ${HOST_SRCDIR}/*.hpp)
HOST_SRCS = $(wildcard ${HOST_SRCDIR}/*.cpp)
HOST_OBJS = $(patsubst $(HOST_SRCDIR)/%.cpp,$(HOST_O_DIR)/%.o,$(HOST_SRCS))
HOST_OBJS += $(HOST_O_DIR)/test_utils.o

CXX := g++-13
CXXFLAGS += -c
CXXFLAGS += -std=c++23
CXXFLAGS += -ggdb
CXXFLAGS += -I/opt/xilinx/xrt/include
CXXFLAGS += -I${HOME_DIR}/host
CXXFLAGS += -I${HOME_DIR}/../../../runtime_lib/test_lib

LDFLAGS += -lm
LDFLAGS += -L/opt/xilinx/xrt/lib
LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
LDFLAGS += -lxrt_coreutil

.PHONY: all link bitstream host clean instructions
all: ${HOST_C_TARGET} ${XCLBIN_MLIR_SRCS} ${XCLBIN_TARGET} ${INSTS_TARGET} 

clean:
	-@rm -rf build
	-@rm -rf log
	-@rm -rf ${HOST_C_TARGET}
	-@rm -rf trace*
	-@rm -rf *.xclbin
	-@rm -rf *.txt


test:
	echo "test"
	echo ${AIEOPT_DIR}



instructions: ${INSTS_TARGET}


kernel: ${KERNEL_OBJS}


link: ${MLIR_TARGET}

mlir:  ${XCLBIN_MLIR_SRCS}
bitstream: ${XCLBIN_TARGET}


host: ${HOST_C_TARGET}


# Build kernels
${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernel/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(DEVICE),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(DEVICE),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif
	mkdir -p ${BITSTREAM_O_DIR}
	cp ${KERNEL_O_DIR}/*.o ${BITSTREAM_O_DIR}

# # # Build python to mlir
${BITSTREAM_O_DIR}/aie_add.mlir: ${HOME_DIR}/aie_add_placed.py
	mkdir -p ${@D}
	python3 $< -d ${DEVICE}  > $@
# Build python to mlir
${BITSTREAM_O_DIR}/aie_mul.mlir: ${HOME_DIR}/aie_mul_placed.py
	mkdir -p ${@D}
	python3 $< -d ${DEVICE}  > $@

# Build xclbin
${XCLBIN_TARGET}: ${HOME_DIR}/${XCLBIN_MLIR_SRCS} ${KERNEL_OBJS}

	cd ${@D} && aiecc.py --aie-generate-xclbin  --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F}  $(<:${HOME_DIR}/kernel/%=../../kernel/%)

	cp ${XCLBIN_TARGET} ./final.xclbin


${BITSTREAM_O_DIR}/insts_mul.txt:  ${HOME_DIR}/build/bitstream/aie_mul.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge \
		--npu-insts-name=${@:${BITSTREAM_O_DIR}/%=%} $(<:${HOME_DIR}/kernel/%=../../kernel/%)
	cp $@ ./
${BITSTREAM_O_DIR}/insts_add.txt:  ${HOME_DIR}/build/bitstream/aie_add.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-npu-insts --no-compile-host --no-xchesscc --no-xbridge \
		--npu-insts-name=${@:${BITSTREAM_O_DIR}/%=%} $(<:${HOME_DIR}/kernel/%=../../kernel/%)
	cp $@ ./

${HOST_C_TARGET}: ${HOST_OBJS}
	mkdir -p ${HOST_O_DIR}
	echo ${HOST_OBJS}
	$(CXX) -o "$@" $(+) $(LDFLAGS)

$(HOST_O_DIR)/test.o: $(HOST_SRCDIR)/test.cpp $(HOST_HEADERS)
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"


$(HOST_O_DIR)/test_utils.o: ${HOME_DIR}/../../../runtime_lib/test_lib/test_utils.cpp
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

.PHONY: run_mul
run_mul: ${HOST_C_TARGET} ${XCLBIN_MLIR_SRCS} ${XCLBIN_TARGET} ${INSTS_TARGET}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 1

.PHONY: run_add
run_add: ${HOST_C_TARGET} ${XCLBIN_MLIR_SRCS} ${XCLBIN_TARGET} ${INSTS_TARGET}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 0

.PHONY: run
run: ${HOST_C_TARGET} ${XCLBIN_MLIR_SRCS} ${XCLBIN_TARGET} ${INSTS_TARGET} 
	echo "run_mul"
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 1
	echo "run_add"
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 0

