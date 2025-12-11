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

include ${srcdir}/../../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = simple_repeat
N = 128 # input size in int32 elements
memtile_repeat_count = 4
data_out_size = $(shell echo $$(($(N) * $(memtile_repeat_count))))

aie_py_src=${targetname}.py

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

.PHONY: all template clean

all: build/final.xclbin

build/aie2.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${N} ${devicename} > $@

build/final.xclbin: build/aie2.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DN=${N} -DMEMTILE_REPEAT_COUNT=${memtile_repeat_count}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -l ${N}

run_py: build/final.xclbin build/insts.bin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -l ${N}

clean:
	rm -rf build _build ${targetname}*.exe