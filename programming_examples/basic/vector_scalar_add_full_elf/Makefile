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

all: build/final.xclbin

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_scalar_add

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/full_elf.elf: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --generate-full-elf --no-compile-host \
		--no-xchesscc --no-xbridge --full-elf-name=full_elf.elf ${<F}

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}.exe build/full_elf.elf
	${powershell} ./$< --elf build/full_elf.elf -k main::sequence

clean:
	rm -rf build _build ${targetname}.exe
