##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_vector_mul
aie_py_src = ${targetname}.py
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all run run_py clean

all:
	@echo "JIT mode: kernel compiles on first 'make run'."

run: run_py

run_py:
	${powershell} python3 ${srcdir}/${aie_py_src} --dev ${devicename}

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
