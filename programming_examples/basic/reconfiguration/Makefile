##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Three run modes, all driven by a single C++ testbench (test.cpp) selected via
# compile-time macros:
#
#   make run           reconfig flow: full-ELF, multi-device aiex.configure/run
#                      (test.cpp -DFULL_ELF).
#   make run_xclbin    single flow: one device (no load_pdi), one run through
#                      the ordinary xclbin + insts testbench (test.cpp).
#   make run_runlist   single flow: the same design executed RECONFIGS times
#                      chained in an xrt::runlist (test.cpp -DRUNLIST).
#
# All design artifacts embed CORES/NOPS/RECONFIGS so changing a parameter never
# picks up a stale artifact.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

# Design parameters. Override on the command line, e.g.
#   make CORES=4 NOPS=64 RECONFIGS=2 run
CORES ?= 1
NOPS ?= 0
RECONFIGS ?= 1

suffix := c${CORES}_n${NOPS}_r${RECONFIGS}

reconfig_elf := build/reconfig_${suffix}.elf
xclbin := build/final_${suffix}.xclbin
insts_bin := build/insts_${suffix}.bin

# Host testbench build (no CMake; compile test.cpp directly against XRT).
XRT_FLAGS := -I${XILINX_XRT}/include -L${XILINX_XRT}/lib -luuid -lxrt_coreutil -lrt -lstdc++

.PHONY: all run run_xclbin run_runlist build_fullelf build_xclbin build_runlist clean

all: build_fullelf

# ---- reconfig flow: full-ELF, multi-device aiex.configure / aiex.run ----
${reconfig_elf}: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow reconfig --cores ${CORES} --nops ${NOPS} --reconfigs ${RECONFIGS} > build/reconfig_${suffix}.mlir
	cd build && aiecc -v --generate-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos reconfig_${suffix}.mlir
	mv build/aie.elf $@

reconfiguration_fullelf.exe: ${srcdir}/test.cpp
	${CXX} -std=c++17 -DFULL_ELF $< -o $@ ${XRT_FLAGS}

build_fullelf: ${reconfig_elf} reconfiguration_fullelf.exe

run: build_fullelf
	./reconfiguration_fullelf.exe ${reconfig_elf} ${CORES}

# ---- single flow: one device (no load_pdi), xclbin + insts ----
${xclbin} ${insts_bin} &: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow single --cores ${CORES} --nops ${NOPS} > build/single_${suffix}.mlir
	cd build && aiecc --no-xchesscc --no-xbridge --aie-generate-xclbin --aie-generate-npu-insts \
		--xclbin-name=$(notdir ${xclbin}) --npu-insts-name=$(notdir ${insts_bin}) \
		--no-compile-host single_${suffix}.mlir

reconfiguration_xclbin.exe: ${srcdir}/test.cpp
	${CXX} -std=c++17 $< -o $@ ${XRT_FLAGS}

reconfiguration_runlist.exe: ${srcdir}/test.cpp
	${CXX} -std=c++17 -DRUNLIST $< -o $@ ${XRT_FLAGS}

build_xclbin: ${xclbin} ${insts_bin} reconfiguration_xclbin.exe

build_runlist: ${xclbin} ${insts_bin} reconfiguration_runlist.exe

run_xclbin: build_xclbin
	./reconfiguration_xclbin.exe ${xclbin} ${insts_bin} ${CORES}

run_runlist: build_runlist
	./reconfiguration_runlist.exe ${xclbin} ${insts_bin} ${CORES} ${RECONFIGS}

clean:
	rm -rf build reconfiguration_fullelf.exe reconfiguration_xclbin.exe reconfiguration_runlist.exe
