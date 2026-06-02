##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Higher-level (@iron.jit) path:
#   make run        — JIT-compile + run + benchmark on the attached NPU.
#   make emit-mlir  — print the lowered MLIR (build/aie.mlir).
#
# Decomposed-XRT walkthrough (same vector_scalar_mul.py source via
# --xclbin-path/--insts-path):
#   make all                — drop build/final.xclbin + build/insts.bin
#                             + myFirstProgram.exe.
#   make run_cpp [N]        — run the C++ test harness; iters/warmup via
#                             extra args (see run-10 / run-10-warmup).
#   make run_py             — run the Python test harness.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../programming_examples/makefile-common

targetname = myFirstProgram
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all run emit-mlir run_cpp run_py run_cpp-10 run_cpp-10-warmup run_py-10 clean

all: ${xclbin_target} ${insts_target} ${targetname}.exe

run:
	${powershell} python3 ${srcdir}/vector_scalar_mul.py --dev ${devicename}

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/vector_scalar_mul.py --dev ${devicename} --emit-mlir > build/aie.mlir

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},vector_scalar_mul.py,-d ${devicename}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

run_cpp: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

run_cpp-10: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE --iters 10

run_cpp-10-warmup: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE --iters 10 --warmup 4

run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} -k MLIR_AIE

run_py-10: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} -k MLIR_AIE --iters 10

clean:
	rm -rf build _build ${targetname}.exe
