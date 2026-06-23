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
# Two multi-column reduce-max variants:
#   * VARIANT=col_wise  (default) — col-wise reduction, 2 cores per col
#   * VARIANT=row_wise            — row-wise reduction
#
# Kernels (reduce_max.cc) come from the iron kernel library; no Makefile
# kernel.o rule needed.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

SHELL := /bin/bash

orig_targetname = vector_reduce_max
in1_size ?= 524288
out_size ?= 4
trace_size ?= 524288
dtype ?= i32
VARIANT ?= col_wise

aie_py_src = ${VARIANT}_${orig_targetname}.py
targetname = ${VARIANT}_${orig_targetname}

ifeq ($(dtype),bf16)
  dtype_cpp = test_utils::bfloat16_t
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

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call jit_xclbin,${trace_xclbin_target},${trace_insts_target},${aie_py_src},${aieargs} -t ${trace_size}))
$(eval $(call build_host_exe,${targetname},_build,../test.cpp,-DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size} -DDTYPE=${dtype_cpp},${srcdir}/..))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: run_py
run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/../test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -i1s ${in1_size} -os ${out_size}

.PHONY: trace
trace: ${targetname}.exe ${trace_xclbin_target} ${trace_insts_target}
	${powershell} ./$< -x ${trace_xclbin_target} -i ${trace_insts_target} -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../../python/utils/trace/parse.py --input trace.txt --mlir $(trace_xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output trace_${targetname}.json
	${srcdir}/../../../../python/utils/trace/get_trace_summary.py --input trace_${targetname}.json

.PHONY: clean_trace
clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

.PHONY: clean
clean:
	rm -rf build _build ${orig_targetname}_*.exe *_${orig_targetname}.exe
