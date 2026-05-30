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
# @iron.jit drives aiecc via --xclbin-path / --insts-path; the library
# ``kernels.conv2dk1`` (uint8 output → fused ReLU) is auto-built into
# the JIT work_dir.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = conv2d_fused_relu
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

width ?= 32
height ?= 32
in_channels ?= 64
out_channels ?= 64

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))

.PHONY: run_py
run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} --kernel MLIR_AIE

.PHONY: clean
clean:
	rm -rf build log
