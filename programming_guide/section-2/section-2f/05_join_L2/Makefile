##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# distribute_and_join_L2.py runs end-to-end on the NPU (`make run` /
# `make emit-mlir`).  join_L2.py is structural-only and is built via
# `make emit-mlir-join` (no NPU run path; no compute to verify).

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
