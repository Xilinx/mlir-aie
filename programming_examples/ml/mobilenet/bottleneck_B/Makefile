#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

mlirFileName = aie
stride = 1
VPATH := ${srcdir}/../../../../aie_kernels/aie2

all: build/conv2dk1_fused_relu.o build/conv2dk3_dw.o build/conv2dk1_ui8.o  build/final.xclbin

build/${mlirFileName}.mlir: ${srcdir}/aie2_stride${stride}.py
	mkdir -p ${@D}
	python3 $< > $@

insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/conv2dk1_fused_relu.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

build/conv2dk3_dw.o: conv2dk3_dw.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR  -DUINT8_ACT -c $< -o $@

build/conv2dk1_ui8.o: conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DSCALAR  -c $< -o ${@F}


build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1_fused_relu.o  build/conv2dk3_dw.o build/conv2dk1_ui8.o 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--basic-alloc-scheme \
		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py: 
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE
