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
# Three single-column reduce-max variants share one design dispatcher.
# Pick one via VARIANT={shared, chained, memtile}; the kernel.o builds
# (reduce_max.cc) come from the iron kernel library via
# @iron.jit + ExternalFunction(source_file=…), so no Makefile-side
# build/reduce_max.cc.o rule.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

SHELL := /bin/bash

orig_targetname = vector_reduce_max
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
in1_size ?= 524288
out_size ?= 4
trace_size ?= 1024
dtype ?= i32
VARIANT ?= shared

targetname = ${orig_targetname}_${VARIANT}
aie_py_src = ${targetname}.py

ifeq ($(dtype),bf16)
  dtype_cpp = std::bfloat16_t
endif
ifeq ($(dtype),i32)
  dtype_cpp = std::int32_t
endif

aieargs = -d ${devicename} -i1s ${in1_size} -os ${out_size} --dtype ${dtype}

xclbin_target       = build/${VARIANT}/final.xclbin
insts_target        = build/${VARIANT}/insts.bin
trace_xclbin_target = build/${VARIANT}/final_trace.xclbin
trace_insts_target  = build/${VARIANT}/insts_trace.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

${xclbin_target} ${insts_target} &: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target}

${trace_xclbin_target} ${trace_insts_target} &: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} -t ${trace_size} --xclbin-path=${trace_xclbin_target} --insts-path=${trace_insts_target}

${targetname}.exe: ${srcdir}/../test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}/..` -DTARGET_NAME=${targetname} -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size} -DDTYPE=${dtype_cpp}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: run_py
run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/../test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -i1s ${in1_size} -os ${out_size}

.PHONY: trace
trace: ${targetname}.exe ${trace_xclbin_target} ${trace_insts_target}
	${powershell} ./$< -x ${trace_xclbin_target} -i ${trace_insts_target} -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../../python/utils/trace/parse.py --input trace.txt --mlir build/${VARIANT}/input_with_addresses.mlir --output trace_${targetname}.json
	${srcdir}/../../../../python/utils/trace/get_trace_summary.py --input trace_${targetname}.json

.PHONY: clean_trace
clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

.PHONY: clean
clean:
	rm -rf build _build ${orig_targetname}_*.exe
