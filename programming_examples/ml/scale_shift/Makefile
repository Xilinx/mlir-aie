##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = scale_shift
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
length ?= 65536

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -l ${length}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin
elf_target    = build/insts.elf

.PHONY: all
all: ${xclbin_target} ${insts_target} ${elf_target}

$(eval $(call jit_xclbin_elf,${xclbin_target},${insts_target},${elf_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
