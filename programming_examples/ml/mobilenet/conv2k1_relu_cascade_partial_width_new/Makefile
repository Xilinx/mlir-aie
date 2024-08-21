#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

mlirFileName = aie2_cascade
VPATH := ${srcdir}/../../../../aie_kernels/aie2

all: build/conv2dk1_put.o build/conv2dk1_get.o build/final.xclbin run_py

build/${mlirFileName}.mlir: ${srcdir}/${mlirFileName}.py
	mkdir -p ${@D}
	python3 $< > $@


build/conv2dk1_put.o: bn_conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DPARTIAL_PUT_I8_CAS_WIDTH_NEW -c $< -o $@

build/conv2dk1_get.o: bn_conv2dk1_relu.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DPARTIAL_GET_I8_CAS_WIDTH_NEW -c $< -o $@

clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim \
		build/insts.txt \
		build/*.log aie_partition.json build/*.bin build/BOOT.BIN _x build/test.exe
		
# clean:
# 	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
# 		chess* *.o insts.txt \
# 		*.log aie_partition.json *.bin BOOT.BIN _x test.exe


insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1_put.o  build/conv2dk1_get.o 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--basic-alloc-scheme \
		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}

run_py: 
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE


# build/final.xclbin: build_new/aie2_opt.mlir build_new/conv2dk1_put.o  build_new/conv2dk1_get.o 
# 	cd build_new && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
# 		--basic-alloc-scheme \
# 		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}

# insts.txt: aie2_opt.mlir
# 	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

# run_py: 
# 	${powershell} python3 ${srcdir}/test.py -x build_new/final.xclbin -i build_new/insts.txt -k MLIR_AIE
