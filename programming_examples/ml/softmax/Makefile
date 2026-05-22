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
# @iron.jit drives aiecc directly via --xclbin-path / --insts-path; the
# library ``kernels.softmax`` is auto-built into the JIT work_dir, so no
# separate Makefile rule for the .o is needed.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = softmax
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
size ?= 262144
chans ?= 2
n_iterations ?= 20
n_warmup ?= 10

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -l ${size} -ch ${chans}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

npu_num = $(if $(filter npu2,$(devicename)),2,1)

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE \
		--npu ${npu_num} --iters ${n_iterations} --warmup ${n_warmup} --size ${size}

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
