#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

include ../../makefile-common

mlirFileName = aieWithTrace_1core

all: build/conv2dk1_i8.o build/final.xclbin


build/${mlirFileName}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@


insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/conv2dk1_i8.o: ../../../aie_kernels/aie2/conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/final.xclbin: build/${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log* *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
run_py: 
	${powershell} python3 test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE