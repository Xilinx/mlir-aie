##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Higher-level path (default): `make run` JIT-compiles + runs the
# @iron.jit design on the attached NPU. `make emit-mlir` prints the
# lowered MLIR without touching the NPU.
#
# Decomposed-XRT walkthrough: `make all` drops the xclbin + insts pair
# + the C++ host exe into build/ (kernel auto-built into the JIT
# cache). `make run_cpp` runs the C++ host harness in test.cpp against
# those artifacts; `make run_py` runs the explicit-NPUKernel /
# DefaultNPURuntime Python harness in test.py.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../programming_examples/makefile-common

targetname = vectorScalar
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all run emit-mlir run_cpp run_py clean

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

run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
