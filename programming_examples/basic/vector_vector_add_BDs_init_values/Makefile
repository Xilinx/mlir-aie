##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# NPU paths use compile_mlir_module directly via --xclbin-path / --insts-path.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = vector_vector_add
col ?= 0

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -c ${col}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
