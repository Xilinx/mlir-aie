##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# @iron.jit handles compile + run end-to-end (`make run`).
# `make emit-mlir` prints the lowered MLIR without touching the NPU.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../../programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all run emit-mlir clean

all: run

run:
	${powershell} python3 ${srcdir}/to_stream.py --dev ${devicename}

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/to_stream.py --dev ${devicename} --emit-mlir > build/aie.mlir

clean:
	rm -rf build
