#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

CHESS ?= false
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

mlirFileName = aieWithTrace_1core

VPATH := ${srcdir}/../../../aie_kernels/aie2

aie_py_src=conv2d_fused_relu.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=conv2d_fused_relu_placed.py
endif

all: build/conv2dk1.o build/final.xclbin

build/${mlirFileName}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/conv2dk1.o: conv2dk1.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F};
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}
endif
else ifeq ($(devicename),npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DINT8_ACT -c $< -o ${@F};
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -DINT8_ACT -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1.o  
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		 --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)
endif

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log* *.xclbin sim \
		chess* *.o insts.bin \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py: build/final.xclbin
	${powershell} python3  ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
