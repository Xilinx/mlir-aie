#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

DEVICE ?= npu1
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


# Kernel stuff
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)
LLVM_AIE_DIR := $(shell realpath $(dir $(shell which aie-opt))/../../)
ifeq ($(DEVICE), npu2) # Strix point
	AIETOOLS_DIR := $(shell realpath $(dir $(shell which xchesscc))/../..)
	AIE_INCLUDE_DIR := ${AIETOOLS_DIR}/data/versal_prod/lib
	AIE2_INCLUDE_DIR := ${AIETOOLS_DIR}/data/aie_ml/lib
else # Hawkeye point
	VITIS_ROOT := $(shell realpath $(dir $(shell which xchesscc))/../..)
	AIETOOLS_DIR := ${VITIS_ROOT}/aietools
	AIE_INCLUDE_DIR := ${VITIS_ROOT}/aietools/data/versal_prod/lib
	AIE2_INCLUDE_DIR := ${VITIS_ROOT}/aietools/data/aie_ml/lib

	CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}
	PEANO_INSTALL_DIR := ${LLVM_AIE_DIR}/llvm-aie
	PEANOWRAP2_FLAGS = -O2 -v -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -I /tools/Xilinx/Vitis/2023.2/aietools/include
endif

CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include 
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include 

CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include -D__AIENGINE__=2 -D__AIEARCH__=20
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}

# Kernel
KERNEL_O_DIR := build/bitstream

KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernel/*.cc)
KERNEL_OBJS := $(patsubst ${HOME_DIR}/kernel/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernel/*.h)

# MLIR stuff
MLIR_O_DIR := build/mlir

BITSTREAM_O_DIR := build/bitstream
XCLBIN_MLIR_SRCS := ${HOME_DIR}/mlir/aie_add.mlir
XCLBIN_MLIR_SRCS += ${HOME_DIR}/mlir/aie_mul.mlir
INSTS_TARGET := ${BITSTREAM_O_DIR}/insts_mul.txt
INSTS_TARGET += ${BITSTREAM_O_DIR}/insts_add.txt
XCLBIN_TARGET := ${BITSTREAM_O_DIR}/final.xclbin

# Host stuff
HOST_SRCDIR := ${HOME_DIR}/host
HOST_O_DIR := build/host
HOST_C_TARGET := host.exe

HOST_HEADERS = $(wildcard ${HOST_SRCDIR}/*.hpp)
HOST_SRCS = $(wildcard ${HOST_SRCDIR}/*.cpp)
HOST_OBJS = $(patsubst $(HOST_SRCDIR)/%.cpp,$(HOST_O_DIR)/%.o,$(HOST_SRCS))
HOST_OBJS += $(HOST_O_DIR)/test_utils.o

CXX := g++-13
ifeq ($(DEVICE),npu1)
	CXXFLAGS += -c
	CXXFLAGS += -std=c++23
	CXXFLAGS += -ggdb
	CXXFLAGS += -I/opt/xilinx/xrt/include
	CXXFLAGS += -I/usr/include/boost
	CXXFLAGS += -I${HOME_DIR}/host
	CXXFLAGS += -I${HOME_DIR}/../../../runtime_lib/test_lib

	LDFLAGS += -lm
	LDFLAGS += -L/opt/xilinx/xrt/lib
	LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
	LDFLAGS += -lxrt_coreutil
	LDFLAGS += -lboost_program_options -lboost_filesystem
else ifeq ($(DEVICE),npu2)
	CXXFLAGS += -c
	CXXFLAGS += -std=c++23
	CXXFLAGS += -ggdb
	CXXFLAGS += -I/opt/xilinx/xrt/include
	CXXFLAGS += -I/usr/include/boost
	CXXFLAGS += -I${HOME_DIR}/host
	CXXFLAGS += -I${HOME_DIR}/../../../runtime_lib/test_lib

	LDFLAGS += -lm
	LDFLAGS += -L/opt/xilinx/xrt/lib
	LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
	LDFLAGS += -lxrt_coreutil
	LDFLAGS += -lboost_program_options -lboost_filesystem
endif

.PHONY: all link bitstream host clean instructions
all: ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}

clean:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf host.exe
	-@rm -rf trace*
	-@rm -rf *.xclbin
	-@rm -rf *.txt


test:
	echo "test"
	echo ${AIEOPT_DIR}



instructions: ${INSTS_TARGET}

kernel: ${KERNEL_OBJS}

link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGET}


host: ${HOST_C_TARGET}


# Build kernels
${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernel/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(DEVICE),npu1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(DEVICE),npu2)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif

# Build bitstream
${XCLBIN_TARGET}: ${XCLBIN_MLIR_SRCS} ${KERNEL_OBJS}
ifeq ($(DEVICE),npu1)
	sed -i 's/npu2/npu1_1col/' $<
else ifeq ($(DEVICE),npu2)
	sed -i 's/npu1_1col/npu2/' $<
else
	echo "Device type not supported"
endif
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host \
		--xclbin-name=${@F} $(<:${HOME_DIR}/kernel/%=../../kernel/%)
	cp ${XCLBIN_TARGET} ./final.xclbin


# Build instructions
${BITSTREAM_O_DIR}/insts_%.txt: ${HOME_DIR}/mlir/aie_%.mlir
ifeq ($(DEVICE),npu1)
	sed -i 's/npu2/npu1_1col/' $<
else ifeq ($(DEVICE),npu2)
	sed -i 's/npu1_1col/npu2/' $<
else
	echo "Device type not supported"
endif
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-only-generate-npu --no-compile-host \
		--npu-insts-name=${@:${BITSTREAM_O_DIR}/%=%} $(<:${HOME_DIR}/kernel/%=../../kernel/%) 
	cp $@ ./

${HOST_C_TARGET}: ${HOST_OBJS}
	mkdir -p ${HOST_O_DIR}
	echo ${HOST_OBJS}
	$(CXX) -o "$@" $(+) $(LDFLAGS)

$(HOST_O_DIR)/host.o: $(HOST_SRCDIR)/host.cpp $(HOST_HEADERS)
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_O_DIR)/%.o: ${HOME_DIR}/../../../runtime_lib/test_lib/%.cpp $(HOST_HEADERS)
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_O_DIR)/test_utils.o: ${HOME_DIR}/../../../runtime_lib/test_lib/test_utils.cpp
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

.PHONY: run_mul
run_mul: ${HOST_C_TARGET} ${XCLBIN_TARGET} ${INSTS_TARGET}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 1

.PHONY: run_add
run_add: ${HOST_C_TARGET} ${XCLBIN_TARGET} ${INSTS_TARGET}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 0

.PHONY: run
run: ${HOST_C_TARGET} ${XCLBIN_TARGET} ${INSTS_TARGET}
	echo "run_mul"
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 1
	echo "run_add"
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	./$< 0

