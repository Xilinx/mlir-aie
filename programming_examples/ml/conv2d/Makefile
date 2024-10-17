#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

mlirFileName = aie

VPATH := ${srcdir}/../../../aie_kernels/aie2

all: build/conv2dk1_i8.o build/final.xclbin

build/conv2dk1_i8.o: conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}

build/${mlirFileName}.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< > $@

build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log* *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe