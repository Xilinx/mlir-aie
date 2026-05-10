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

targetname = vector_vector_mul
aie_py_src = ${targetname}.py

.PHONY: all run run_py vck5000 clean

all:
	@echo "JIT mode: kernel compiles on first 'make run'."
	@echo "For VCK5000, run 'make vck5000'."

run: run_py

run_py:
	${powershell} python3 ${srcdir}/${aie_py_src}

# ----------------------------------------------------------------------------
# VCK5000 path: emit MLIR for XCVC1902, build host via aiecc + test_vck5000.cpp.
# Unlike the NPU path, VCK5000 does not go through the @iron.jit host runtime.
# ----------------------------------------------------------------------------

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --emit-mlir-vck5000 > $@

vck5000: build/aie.mlir
	aiecc --xchesscc --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie.mlir \
		-I/opt/xaiengine/include \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		-L/opt/xaiengine/lib \
		-L/lib/x86_64-linux-gnu/ \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,-R/opt/xaiengine/lib \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o test.elf

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
