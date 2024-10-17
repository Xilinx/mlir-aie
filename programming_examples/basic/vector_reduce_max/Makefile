##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

ACDC_AIE = $(dir $(shell which aie-opt))/..

targetname = reduce_max
devicename = npu
col = 0
CHESS_FLAGS=${CHESSCCWRAP2_FLAGS}

all: build/final.xclbin build/insts.txt

VPATH := ${srcdir}/../../../aie_kernels/aie2

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} > $@

build/final.xclbin: build/aie.mlir build/reduce_max.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

trace:
	../../utils/parse_eventIR.py --filename trace.txt --mlir build/aie.mlir --colshift 1 > parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt

clean: clean_trace
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
