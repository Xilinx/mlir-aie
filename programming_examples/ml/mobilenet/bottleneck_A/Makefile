#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

VPATH := ${srcdir}/../../../../aie_kernels/aie2

all: build/final_bn_1.xclbin

build/aie2_%.mlir: ${srcdir}/aie2_%.py ${srcdir}/aie2_bottleneckA.py
	mkdir -p ${@D}
	python3 $< > $@

# insts_.txt: build/${mlirFileName}.mlir
# 	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/conv2dk1_fused_relu.o: conv2dk1.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

build/conv2dk3_dw_stride2.o: conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DSCALAR -DSTRIDE2 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DUINT8_ACT -DSTRIDE2 -c $< -o $@

build/conv2dk3_dw_stride1.o: conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DSCALAR -c $< -o ${@F}

build/conv2dk1_skip.o: bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS}  -DSCALAR -c $< -o ${@F}

build/conv2dk1_i8.o: conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS}  -DSCALAR -c $< -o ${@F}

build/combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a: build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride2.o build/conv2dk1_i8.o  
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a: build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride1.o build/conv2dk1_i8.o  
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a:  build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride1.o build/conv2dk1_skip.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/final_%.xclbin build/inst_%.txt &: build/aie2_%.mlir build/combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--basic-alloc-scheme \
		--xclbin-name=${@F} --npu-insts-name=insts_$*.txt ${<F}

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py_1: build/final_bn_1.xclbin build/insts_bn_1.txt build/aie2_bn_1.mlir
	${powershell} python3 ${srcdir}/test_bn_1.py -x build/final_bn_1.xclbin -i build/insts_bn_1.txt -k MLIR_AIE

run_py_2: build/final_bn_2.xclbin build/insts_bn_2.txt build/insts_bn_2.txt
	${powershell} python3 ${srcdir}/test_bn_2.py -x build/final_bn_2.xclbin -i build/insts_bn_2.txt -k MLIR_AIE
