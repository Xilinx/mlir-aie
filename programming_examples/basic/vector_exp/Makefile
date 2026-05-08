##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_exp
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

aie_py_src=${targetname}.py

.PHONY: all run clean

all:
	@echo "JIT mode: kernel compiles on first 'make run'."

run:
	${powershell} python3 ${srcdir}/${aie_py_src}

clean:
	rm -rf build _build ${targetname}*.exe
