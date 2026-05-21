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
# The vck5000 (Versal AIE1) path uses the legacy print-MLIR + aiecc flow.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = passthrough_dmas
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
LENGTH ?= 4096

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -n ${LENGTH}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

${xclbin_target} ${insts_target} &: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target}

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -l ${LENGTH}

# vck5000 (Versal AIE1) path stays on the legacy print-MLIR + aiecc flow.
# Three topologies share the design body, picked via PLIO=none|input|output:
#   * vck5000              — non-PLIO (shim → memtile → shim)
#   * vck5000_plio_input   — PLIO ObjectFifo on input side
#   * vck5000_plio_output  — PLIO ObjectFifo on output side

build/aie_vck5000.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d xcvc1902 -n ${LENGTH} --plio none --emit-mlir > $@

build/aie_vck5000_plio_input.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d xcvc1902 -n ${LENGTH} --plio input --emit-mlir > $@

build/aie_vck5000_plio_output.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d xcvc1902 -n ${LENGTH} --plio output --emit-mlir > $@

define vck5000_link
	aiecc --link_against_hsa --host-target=x86_64-amd-linux-gnu $1 \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		-L/lib/x86_64-linux-gnu/ \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o $2
endef

.PHONY: vck5000
vck5000: build/aie_vck5000.mlir
	$(call vck5000_link,$<,test.elf)

.PHONY: vck5000_plio_input
vck5000_plio_input: build/aie_vck5000_plio_input.mlir
	$(call vck5000_link,$<,input.elf)

.PHONY: vck5000_plio_output
vck5000_plio_output: build/aie_vck5000_plio_output.mlir
	$(call vck5000_link,$<,output.elf)

.PHONY: run_vck5000
run_vck5000:
	./test.elf

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf input.elf output.elf ${targetname}.exe
