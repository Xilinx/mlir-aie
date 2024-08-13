#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common
# mlirFileName=aie2_bn10_bn11



# mlirFileName=aie2_bn10_bn11_bn12


# mlirFileName = aie2_bottleneckB
# testFileName=test


# mlirFileName=aie2_bn10
# testFileName=test_bn10

# mlirFileName=aie2_bn10_bn11_no_skip #make sure to update the kernel accordingly
# testFileName=test_bn10_bn11_no_skip



# mlirFileName = aie2_bottleneckB
# testFileName=test
mlirFileName=aie2_bn10_bn11
testFileName=test_bn10_bn11


mlirFileName=aie2_bn10_bn11_bn12
testFileName=test_bn10_bn11_bn12

testFileName=test
mlirFileName = aie2_bottleneckB



# mlirFileName = aie2_bottleneckB_wts_L2
# testFileName=test_wts_L2

VPATH := ${srcdir}/../../../../aie_kernels/aie2

all: build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
	build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
	build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
	build/final.xclbin run_py

build/${mlirFileName}.mlir: ${srcdir}/${mlirFileName}.py
	mkdir -p ${@D}
	python3 $< > $@

insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/bn10_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN10 -DINT8_ACT -c $< -o $@

build/bn10_conv2dk3_dw.o: bn_conv2dk3_dw.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN10 -DSTRIDE1   -c $< -o $@

build/bn10_conv2dk1_ui8.o: bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBN10 -DSCALAR  -c $< -o ${@F}
# bn11
build/bn11_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN11 -DINT8_ACT -c $< -o $@

build/bn11_conv2dk3_dw.o: bn_conv2dk3_dw.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN11 -DSTRIDE1  -c $< -o $@

build/bn11_conv2dk1_skip.o: bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBN11 -DSCALAR  -c $< -o ${@F}
# bn12
build/bn12_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN12 -DINT8_ACT -c $< -o $@

build/bn12_conv2dk3_dw_stride2.o: bn_conv2dk3_dw.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DSTRIDE2  -DBN12 -c $< -o $@

build/bn12_conv2dk1_ui8.o: bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBN12 -DSCALAR  -c $< -o ${@F}

build/final.xclbin: build/${mlirFileName}.mlir build/bn10_conv2dk1_fused_relu.o  build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o \
						build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
						build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--basic-alloc-scheme \
		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}

# clean:
# 	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
# 		chess* *.o insts.txt \
# 		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim \
		build/insts.txt \
		build/*.log aie_partition.json build/*.bin build/BOOT.BIN _x build/test.exe

run_py: 
	${powershell} python3 ${srcdir}/${testFileName}.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE
