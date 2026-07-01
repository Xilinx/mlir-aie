##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# @iron.jit handles compile + run end-to-end (`make run`).
# `make emit-mlir` prints the lowered MLIR without touching the NPU
# (writes to build/aie.mlir).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../programming_examples/makefile-common

.PHONY: all run emit-mlir clean

all: run

run:
	${powershell} python3 ${srcdir}/aie2.py --dev ${devicename}

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/aie2.py --dev ${devicename} --emit-mlir > build/aie.mlir

clean:
	rm -rf build
