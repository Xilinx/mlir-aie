##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# `make run` / `make run-multi` JIT-compile + run the single-core /
# multi-core variants on the attached NPU.
# `make emit-mlir` / `make emit-mlir-multi` print the lowered MLIR
# without touching the NPU.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all run run-multi emit-mlir emit-mlir-multi clean

all: run run-multi

run:
	${powershell} python3 ${srcdir}/aie2.py --dev ${devicename}

run-multi:
	${powershell} python3 ${srcdir}/aie2_multi.py --dev ${devicename}

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/aie2.py --dev ${devicename} --emit-mlir > build/aie.mlir

emit-mlir-multi:
	mkdir -p build
	${powershell} python3 ${srcdir}/aie2_multi.py --dev ${devicename} --emit-mlir > build/aie_multi.mlir

clean:
	rm -rf build
