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
# @iron.jit drives aiecc via --xclbin-path / --insts-path; the
# ExternalFunction for ``conv2dk14_i8`` (from aie_kernels/aie2p) is
# auto-built into the JIT work_dir. This design is aie2p (npu2) only.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = conv2dk14
devicename ?= npu2

width ?= 896
height ?= 896
num_act ?= 8

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -wd ${width} -ht ${height}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))

.PHONY: run_py
run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE \
		-wd ${width} -ht ${height} -ic 4 -oc 1152 -ksz 14 -na ${num_act}

.PHONY: clean
clean:
	rm -rf build log
