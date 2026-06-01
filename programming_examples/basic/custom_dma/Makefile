##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/generic

devicename ?= npu2
targetname = custom_dma

.PHONY: all clean

all: build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

build/aie.mlir: ${srcdir}/${targetname}.py
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/final.xclbin: build/aie.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build
