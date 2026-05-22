##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# NPU paths use @iron.jit's compile() directly via --xclbin-path / --insts-path.
# The vck5000 (Versal AIE1) path uses the legacy print-MLIR + aiecc flow
# because aiecc.py's --xchesscc + --link_against_hsa toolchain is not driven
# through @iron.jit.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_vector_modulo
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

aie_py_src = ${targetname}.py
aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

# vck5000 (Versal AIE1) path stays on the legacy print-MLIR + aiecc flow.
ACDC_AIE = $(dir $(shell which aie-opt))/..

build/aie_vck5000.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d xcvc1902 --emit-mlir > $@

.PHONY: vck5000
vck5000: build/aie_vck5000.mlir
	aiecc --xchesscc --link_against_hsa --host-target=x86_64-amd-linux-gnu $< \
		-I/opt/xaiengine/include \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		-L/opt/xaiengine/lib \
		-L/lib/x86_64-linux-gnu/ \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,-R/opt/xaiengine/lib \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o test.elf

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
