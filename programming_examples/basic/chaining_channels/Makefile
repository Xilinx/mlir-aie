##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Low-level placed IRON design (see chaining_channels.py docstring).
# The script's --xclbin-path / --insts-path mode drives
# compile_mlir_module directly, so the Makefile shape matches the
# @iron.jit ports.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = chaining_channels
devicename ?= npu2
LENGTH ?= 1024
TRACE ?= 0

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -n ${LENGTH} -t ${TRACE}

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
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -l ${LENGTH} -t ${TRACE}

.PHONY: parse_trace
parse_trace: run
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/input_with_addresses.mlir --output trace_${targetname}.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_${targetname}.json

.PHONY: clean_trace
clean_trace:
	rm -rf trace.txt trace_${targetname}.json

.PHONY: clean
clean: clean_trace
	rm -rf build _build inst aie.mlir.prj core_* ${targetname}.exe
