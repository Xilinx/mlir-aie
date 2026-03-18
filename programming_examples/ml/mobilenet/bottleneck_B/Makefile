#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common
VPATH := ${srcdir}/../../../../aie_kernels/aie2

mlirFileName = aie2_bottleneckB
testSource = test_bottleneckB

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
ifeq (${devicename}, npu)
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2_FLAGS}
else
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2P_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2P_FLAGS}
endif 

CHESS ?= false
colshift ?= $(if $(filter npu,$(devicename)),1,0)

all: build/final.xclbin

build/${mlirFileName}.mlir: ${srcdir}/${mlirFileName}.py
	mkdir -p ${@D}
	python3 $< > $@

# ************************************** bottlneck 10 **************************************
build/bn10_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}
endif 

build/bn10_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DSCALAR -DBN10 -DSTRIDE1   -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS}  -DSCALAR -DBN10 -DSTRIDE1 -c $< -o ${@F}
endif 

build/bn10_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DBN10 -DSCALAR  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN10 -DSCALAR -c $< -o ${@F}
endif 

# ************************************** bottlneck 11 **************************************
build/bn11_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}
endif 

build/bn11_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DSCALAR -DBN11 -DSTRIDE1  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DBN11 -DSTRIDE1 -c $< -o ${@F}
endif 

build/bn11_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DBN11 -DSCALAR  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN11 -DSCALAR -c $< -o ${@F}
endif 

# ************************************** bottlneck 12 **************************************
build/bn12_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}
endif 

build/bn12_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DSCALAR -DSTRIDE2  -DBN12 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DSTRIDE2  -DBN12 -c $< -o ${@F}
endif 

build/bn12_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS}  -DBN12 -DSCALAR  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN12 -DSCALAR -c $< -o ${@F}
endif 

# ************************************** bottleneck C **************************************
build/final.xclbin: build/${mlirFileName}.mlir build/bn10_conv2dk1_fused_relu.o  build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o \
						build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
						build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o 
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)
else
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)
endif

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__

run_py: build/final.xclbin ${srcdir}/${mlirFileName}.py
	${powershell} python3 ${srcdir}/${testSource}.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
