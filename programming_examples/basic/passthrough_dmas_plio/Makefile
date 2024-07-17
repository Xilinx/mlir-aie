##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= xcvc1902
col ?= 26
targetname = passThroughDMAs
LENGTH ?= 1024

all: input output

build/aie-input-plio.mlir: ${srcdir}/aie2-input-plio.py
	mkdir -p ${@D}
	python3 $< ${LENGTH} ${devicename} ${col} > $@

build/aie-output-plio.mlir: ${srcdir}/aie2-output-plio.py
	mkdir -p ${@D}
	python3 $< ${LENGTH} ${devicename} ${col} > $@

input: build/aie-input-plio.mlir
	aiecc.py --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie-input-plio.mlir \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		-L/lib/x86_64-linux-gnu/ \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o input.elf

output: build/aie-output-plio.mlir
	aiecc.py --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie-output-plio.mlir \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		-L/lib/x86_64-linux-gnu/ \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o output.elf

run_vck5000:
	test.elf

clean:
	rm -rf build aie-output-plio.mlir.prj aie-input-plio.mlir.prj core_* input.elf output.elf 
