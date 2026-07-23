##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
runlist_hrx_targetname = vector_scalar_add_runlist_hrx

aie_py_src = ${targetname}.py
aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin
elf_target = build/insts.elf

.PHONY: all
all: ${xclbin_target} ${insts_target} ${elf_target}

$(eval $(call jit_xclbin_elf,${xclbin_target},${insts_target},${elf_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))
$(eval $(call build_host_exe,${runlist_targetname},_build_runlist,test_runlist.cpp,-DTEST_SRC=test_runlist.cpp))
# HRX chain (runlist) testbench. Self-selects the HRX backend (-DUSE_HRX=ON) so
# `make run_runlist_hrx` works without NPU_RUNTIME=hrx.
$(eval $(call build_host_exe,${runlist_hrx_targetname},_build_runlist_hrx,test_runlist_hrx.cpp,-DTEST_SRC=test_runlist_hrx.cpp -DUSE_HRX=ON))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE

.PHONY: run_runlist
run_runlist: ${runlist_targetname}.exe ${xclbin_target} ${elf_target}
	${powershell} ./$< -x ${xclbin_target} -i ${elf_target} -k MLIR_AIE

.PHONY: run_runlist_hrx
run_runlist_hrx: ${runlist_hrx_targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: clean
clean:
	rm -rf build _build _build_runlist _build_runlist_hrx ${targetname}.exe ${runlist_targetname}.exe ${runlist_hrx_targetname}.exe
