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
# NPU paths use @iron.jit's compile() directly via --xclbin-path /
# --insts-path / --elf-path.  insts.elf is the ELF-wrapped form the C++
# testbench (xrt::elf + xrt::module flow) loads.
# The same xclbin/insts pair is exercised by two testbenches:
#   * test.cpp          — single run (`make run`)
#   * test_runlist.cpp  — chained runs via xrt::runlist (`make run_runlist`,
#                          NPU2 only — runlist isn't implemented on Phoenix)

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = vector_scalar_add
runlist_targetname = vector_scalar_add_runlist
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

aie_py_src = ${targetname}.py
aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin
elf_target = build/insts.elf

.PHONY: all
all: ${xclbin_target} ${insts_target} ${elf_target}

${xclbin_target} ${insts_target} ${elf_target} &: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target} --elf-path=${elf_target}

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

${runlist_targetname}.exe: ${srcdir}/test_runlist.cpp
	rm -rf _build_runlist
	mkdir -p _build_runlist
	cd _build_runlist && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${runlist_targetname} -DTEST_SRC=test_runlist.cpp
	cd _build_runlist && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build_runlist/${runlist_targetname}.exe $@
else
	cp _build_runlist/${runlist_targetname} $@
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE

.PHONY: run_runlist
run_runlist: ${runlist_targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE

.PHONY: clean
clean:
	rm -rf build _build _build_runlist ${targetname}.exe ${runlist_targetname}.exe
