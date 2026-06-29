# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../../programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all emit-mlir clean

all: emit-mlir

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/distribute_L2.py --dev ${devicename} --emit-mlir > build/aie.mlir

clean:
	rm -rf build
