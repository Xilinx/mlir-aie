#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common
# mlirFileName=aie2_bn10_bn11

mlirFileName = aie2_bottleneckB

# mlirFileName=aie2_bn10_bn11_bn12
# mlirFileName=aie2_bn10



VPATH := ${srcdir}/../../../../aie_kernels/aie2


all: build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o\
	build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
	build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
	build/final.xclbin run_py

build/${mlirFileName}.mlir: ${srcdir}/${mlirFileName}.py
	mkdir -p ${@D}
	python3 $< > $@

#insts.bin: build/${mlirFileName}.mlir
#	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/bn10_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}

build/bn10_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR -DBN10 -DSTRIDE1   -c $< -o ${@F}

build/bn10_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DBN10 -DSCALAR  -c $< -o ${@F}

# bn11
build/bn11_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}

build/bn11_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR -DBN11 -DSTRIDE1  -c $< -o ${@F}

build/bn11_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DBN11 -DSCALAR  -c $< -o ${@F}
# bn12
build/bn12_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}

build/bn12_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR -DSTRIDE2  -DBN12 -c $< -o ${@F}

build/bn12_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DBN12 -DSCALAR  -c $< -o ${@F}

build/fused_bn12_layer2_3.a: build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o
	mkdir -p ${@D}
	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) 

build/final.xclbin: build/${mlirFileName}.mlir build/bn10_conv2dk1_fused_relu.o  build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o \
						build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
						build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o build/fused_bn12_layer2_3.a
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__

run_py: build/final.xclbin ${srcdir}/${mlirFileName}.py
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
