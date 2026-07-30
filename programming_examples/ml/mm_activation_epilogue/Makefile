##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# aie2p (npu2) only: mm_activation_epilogue_row lives under aie_kernels/aie2p/
# and has no aie2 counterpart, so makefile-common's default of `npu` would fail
# to compile the kernel. Must be set BEFORE the include, per that file's own
# contract -- it uses `?=`, so a later assignment here would be a no-op.
devicename ?= npu2

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = mm_activation_epilogue
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
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -l ${length}

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
