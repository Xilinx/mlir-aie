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

include ${srcdir}/../../../makefile-common

targetname = resnet

.PHONY: all run run_py clean

all:
	@echo "JIT mode: kernels compile on first 'make run'."

run: run_py

run_py:
	${powershell} python3 ${srcdir}/test.py

clean:
	rm -rf build _build log ${targetname}*.exe
