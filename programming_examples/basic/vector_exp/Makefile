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

# JIT-mode design: vector_exp.py uses iron.get_current_device() to
# auto-detect the attached NPU at JIT time, so no -d / devicename
# flag is threaded through.

aie_py_src = ${targetname}.py

.PHONY: all run clean

all:
	@echo "JIT mode: kernel compiles on first 'make run'."

run:
	${powershell} python3 ${srcdir}/${aie_py_src}

clean:
	rm -rf build _build ${targetname}*.exe
