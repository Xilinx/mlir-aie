##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Structural-only example: emit-MLIR is the intended use (no NPU run path
# because there's no compute to verify).

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
