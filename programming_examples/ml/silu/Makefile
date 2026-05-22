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
# @iron.jit drives aiecc directly via --xclbin-path / --insts-path; the
# library ``kernels.silu`` is auto-built into the JIT work_dir (LUT-based
# variant on aie2 wraps silu.cc + lut_based_ops.cpp in one source-string
# compile; aie2p uses silu.cc directly).  No separate Makefile rule for
# the .o / archive is needed.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = silu
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
length ?= 65536
chans ?= 2

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -l ${length} -ch ${chans}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -l ${length}

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
