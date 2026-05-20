##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Drives the unified @iron.jit dma_transpose.py: the .py emits the xclbin/
# insts directly to the paths requested below.  Host code (test.cpp) consumes
# those artifacts.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = dma_transpose
M ?= 64
K ?= 32

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -M ${M} -K ${K}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

${xclbin_target} ${insts_target} &: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target}

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

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M ${M} -K ${K}

.PHONY: generate_access_map
generate_access_map:
	python3 ${srcdir}/${aie_py_src} ${aieargs} --generate-access-map

.PHONY: clean
clean:
	rm -rf build _build inst ${targetname}.exe
