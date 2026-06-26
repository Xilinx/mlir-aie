# Copyright (C) 2018-2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../../programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all run emit-mlir emit-mlir-join clean

all: run

run:
	${powershell} python3 ${srcdir}/distribute_and_join_L2.py --dev ${devicename}

emit-mlir:
	mkdir -p build
	${powershell} python3 ${srcdir}/distribute_and_join_L2.py --dev ${devicename} --emit-mlir > build/aie.mlir

emit-mlir-join:
	mkdir -p build
	${powershell} python3 ${srcdir}/join_L2.py --dev ${devicename} --emit-mlir > build/aie_join.mlir

clean:
	rm -rf build
