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

# ---

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt

# ---
UTILS_INCLUDE := -I$(srcdir)/../../../runtime_lib/test_lib/
UTILS_LIB=$(srcdir)/../../../runtime_lib/test_lib/test_utils.cpp
XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

XRT_FLAGS=-I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}
XRT_LIBS=-lxrt_coreutil
CXX=g++-13 -ggdb 

mlir_aie_add?=build/aie_add.mlir
mlir_aie_mul?=build/aie_mul.mlir
add_xclbin_target?=build/add.xclbin
mult_xclbin_target?=build/mult.xclbin
add_insts_target?=build/add_insts.bin
mult_insts_target?=build/mult_insts.bin

host_target?=build/test

DEVICE ?= $(if $(filter 1,$(NPU2)),npu2,npu)

aie_add_placed_py=aie_add_placed.py
aie_mul_placed_py=aie_mul_placed.py

.PHONY: all
all: ${add_xclbin_target} ${mult_xclbin_target} ${host_target}

build/add_mul.o: ${srcdir}/add_mul.cc
	mkdir -p ${@D}
ifeq (${DEVICE}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_m=$m -DDIM_n=$n -c $< -o ${@F}
endif

${mlir_aie_add}: ${srcdir}/${aie_add_placed_py}
	mkdir -p ${@D}
	python3 $< -d ${DEVICE} > $@

${mlir_aie_mul}: ${srcdir}/${aie_mul_placed_py}
	mkdir -p ${@D}
	python3 $< -d ${DEVICE} > $@



${add_xclbin_target}: ${mlir_aie_add}  build/add_mul.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=${add_insts_target:build/%=%} ${<:%=../%}

${mult_xclbin_target}: ${mlir_aie_mul} build/add_mul.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=${mult_insts_target:build/%=%} ${<:%=../%}


${host_target}: ${srcdir}/test.cpp ${mult_xclbin_target} ${add_xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB}  -o $@ $< ${XRT_LIBS}

.PHONY: run_mul run_add run
run_mul: all
	./$(host_target) 1 ${mult_insts_target} ${mult_xclbin_target}
run_add: all
	./$(host_target) 0 ${add_insts_target} ${add_xclbin_target}
run: run_mul run_add

.PHONY: clean
clean:
	-rm -rf build