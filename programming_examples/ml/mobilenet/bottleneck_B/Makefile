#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common
VPATH := ${srcdir}/../../../../aie_kernels/aie2

MLIR_FILE = aie2_bottleneckB
TEST_FILE = test_bottleneckB

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
ifeq (${devicename}, npu)
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2_FLAGS}
else
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2P_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2P_FLAGS}
endif 

CHESS ?= false

ifeq ($(CHESS), true)
	COMPILE_WRAPPER := xchesscc_wrapper ${LOCAL_CHESS_FLAGS}
else
	COMPILE_WRAPPER := ${PEANO_INSTALL_DIR}/bin/clang++ ${LOCAL_PEANO_FLAGS}
endif

colshift ?= $(if $(filter npu,$(devicename)),1,0)

all: build/final.xclbin

build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py
	mkdir -p ${@D}
	python3 $< > $@

# ************************************** bottlneck 10 **************************************
build/bn10_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}

build/bn10_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN10 -DSTRIDE1   -c $< -o ${@F}

build/bn10_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DBN10 -DSCALAR  -c $< -o ${@F}

# ************************************** bottlneck 11 **************************************
build/bn11_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}

build/bn11_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN11 -DSTRIDE1  -c $< -o ${@F}

build/bn11_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DBN11 -DSCALAR  -c $< -o ${@F}

# ************************************** bottlneck 12 **************************************
build/bn12_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}

build/bn12_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DSTRIDE2  -DBN12 -c $< -o ${@F}

build/bn12_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DBN12 -DSCALAR  -c $< -o ${@F}

# ************************************** bottleneck C **************************************
build/final.xclbin: build/${MLIR_FILE}.mlir build/bn10_conv2dk1_fused_relu.o  build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o \
						build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
						build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o 
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)
endif

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__

run_py: build/final.xclbin ${srcdir}/${MLIR_FILE}.py
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
