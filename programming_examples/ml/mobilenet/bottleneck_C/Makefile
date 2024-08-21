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

pythonSource=aie2_bn13_layer2_with_output_split
testSource=test_layer2

pythonSource=aie2_bn13bn14

pythonSource=aie2_bottleneckC
testSource=test

OBJ = build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o 


all: $(OBJ)  build/final.xclbin run_py

# ************************************** bottlneck 13 **************************************

build/bn13_1_conv2dk1_put.o: bn_conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN13_1_PARTIAL_PUT_I8_CAS_WIDTH -c $< -o $@

build/bn13_1_conv2dk1_get.o: bn_conv2dk1_relu.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN13_1_PARTIAL_GET_I8_CAS_WIDTH -c $< -o $@

build/bn13_conv2dk3_dw.o: bn_conv2dk3_dw.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN13 -DSTRIDE1_OUT_SPLIT -c $< -o $@

#  use when we don't split output of conv 3x3
# build/bn13_conv2dk3_dw.o: bn_conv2dk3_dw.cc
# 	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN13 -DSTRIDE1 -c $< -o $@


build/bn13_conv2dk1_put.o: bn_conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH -c $< -o $@

# build/conv2dk1_get.o: bn_conv2dk1_relu.cc
# 	xchesscc -d ${CHESSCC2_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH -c $< -o $@

build/bn13_conv2dk1_skip_get.o: bn_conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH -c $< -o $@

# ************************************** bottlneck 14 **************************************
build/bn14_1_conv2dk1_put.o: bn_conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN14_1_PARTIAL_PUT_I8_CAS_WIDTH -c $< -o $@

build/bn14_1_conv2dk1_get.o: bn_conv2dk1_relu.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN14_1_PARTIAL_GET_I8_CAS_WIDTH -c $< -o $@

build/bn14_conv2dk3_dw.o: bn_conv2dk3_dw.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DBN14 -DSTRIDE1_OUT_SPLIT -c $< -o $@


build/bn14_conv2dk1_put.o: bn_conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH -c $< -o $@

build/bn14_conv2dk1_skip_get.o: bn_conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH -c $< -o $@

build/${mlirFileName}.mlir: ${srcdir}/${pythonSource}.py
	mkdir -p ${@D}
	python3 $< > $@

insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/final.xclbin: build/${mlirFileName}.mlir $(OBJ)
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
	${powershell} python3 ${srcdir}/${testSource}.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

