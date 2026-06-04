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
# conv2dk14.cc kernel is auto-built into the JIT work_dir.
#
# Two variants live in conv2dk14.py:
#   default:                      single-core
#   make ... multi=1              32-core (8 cols x 4 rows)
#
# Both are aie2p (npu2) only.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = conv2dk14
devicename ?= npu2

width ?= 896
height ?= 896
num_act ?= 8
multi ?= 0

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -wd ${width} -ht ${height} $(if $(filter 1,$(multi)),--multi,)

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))

.PHONY: run_py
# -ic / -oc / -ksz are tied to the conv2dk14.cc kernel layer and are
# hardcoded here; only -wd / -ht / -na / multi are tunable via make.
run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} --kernel MLIR_AIE \
		-wd ${width} -ht ${height} -ic 4 -oc 1152 -ksz 14 -na ${num_act}

.PHONY: clean
clean:
	rm -rf build log
