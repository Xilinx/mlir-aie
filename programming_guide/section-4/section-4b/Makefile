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
# @iron.jit path:
#   make run        — JIT-compile + run + verify on the attached NPU.
#   make emit-mlir  — print the lowered MLIR (build/aie.mlir).
#
# Trace path (rt.enable_trace embedded in the runtime sequence):
#   make all-trace            — drop trace-enabled xclbin + insts pair
#                               into build/trace_<targetname>.* for
#                               the explicit-XRT trace walkthrough.
#   make trace_cpp / trace_py — run the harness, parse trace.txt,
#                               produce trace_4b.json + a trace summary.
#
# Decomposed-XRT walkthrough (no trace):
#   make all                  — drop build/final.xclbin + build/insts.bin
#                               + the C++ test harness exe.
#   make run_cpp / run_py     — run the harness against the artifacts.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../programming_examples/makefile-common

targetname    = aie2b
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
in1_size      = 16384
in2_size      = 4
out_size      = ${in1_size}
trace_size    = 8192

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

trace_xclbin_target = build/trace_${targetname}.xclbin
trace_insts_target  = build/trace_${targetname}_insts.bin

.PHONY: all run emit-mlir run_cpp run_py all-trace trace_cpp trace_py clean clean_trace

all: ${xclbin_target} ${insts_target} ${targetname}.exe

run:
	${powershell} python3 ${srcdir}/vector_scalar_mul.py --dev ${devicename}

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/vector_scalar_mul.py --dev ${devicename} --emit-mlir > build/aie.mlir

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},vector_scalar_mul.py,-d ${devicename}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp,-DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}))

run_cpp: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} -k MLIR_AIE --in1-size ${in1_size} --in2-size ${in2_size} --out-size ${out_size}

all-trace: ${trace_xclbin_target} ${trace_insts_target} ${targetname}.exe

$(eval $(call jit_xclbin,${trace_xclbin_target},${trace_insts_target},vector_scalar_mul.py,-d ${devicename} -t ${trace_size}))

trace_cpp: ${targetname}.exe ${trace_xclbin_target} ${trace_insts_target}
	${powershell} ./$< -x ${trace_xclbin_target} -i ${trace_insts_target} -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/input_with_addresses.mlir --output trace_4b.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_4b.json

trace_py: ${trace_xclbin_target} ${trace_insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${trace_xclbin_target} --instr ${trace_insts_target} -k MLIR_AIE -t ${trace_size} --in1-size ${in1_size} --in2-size ${in2_size} --out-size ${out_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/input_with_addresses.mlir --output trace_4b.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_4b.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
