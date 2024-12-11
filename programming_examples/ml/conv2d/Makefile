#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

mlirFileName = aie

trace_size = 16384

VPATH := ${srcdir}/../../../aie_kernels/aie2

aie_py_src=conv2d.py
use_alt?=0

ifeq (${use_alt}, 1)
aie_py_src=conv2d_alt.py
endif

all: build/conv2dk1_i8.o build/final.xclbin

build/conv2dk1_i8.o: conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}

build/${mlirFileName}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< > $@

build/${mlirFileName}_trace.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${trace_size} > $@

build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

build/final_trace.xclbin: build/${mlirFileName}_trace.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_trace.txt $(<:%=../%)

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

trace_py: build/final_trace.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts_trace.txt -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename log/trace_conv2d.txt --mlir build/aie_trace.mlir --colshift 1 > log/trace_conv2d.json

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log* *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe