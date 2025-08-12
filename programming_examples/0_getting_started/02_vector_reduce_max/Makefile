##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

VPATH := ${srcdir}/../../../../aie_kernels/aie2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = aie
.PHONY: all template clean

all: build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

build/aie.mlir: ${srcdir}/aie.py
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/final.xclbin: build/aie.mlir build/reduce_max.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/../test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}/..` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE 

clean:
	rm -rf build _build ${targetname}*.exe