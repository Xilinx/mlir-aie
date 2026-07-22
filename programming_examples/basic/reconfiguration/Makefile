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


# ---
# Design parameters
# ---

CORES ?= 1
NOPS ?= 0
RECONFIGS ?= 1

suffix := c${CORES}_n${NOPS}_r${RECONFIGS}

elf_loadpdi := build/elf_loadpdi_${suffix}.elf
elf_blockwrites := build/elf_blockwrites_${suffix}.elf
elf_ctrlpkt := build/elf_ctrlpkt_${suffix}.elf
xclbin := build/final_${suffix}.xclbin
insts_bin := build/insts_${suffix}.bin


# ---
# Phony
# ---

.PHONY: all run run_xclbin run_runlist build_fullelf build_xclbin build_runlist clean

all: build_fullelf_loadpdi build_fullelf_blockwrites build_fullelf_ctrlpkt build_xclbin build_runlist

build_fullelf_loadpdi: ${elf_loadpdi} test_fullelf

build_fullelf_blockwrites: ${elf_blockwrites} test_fullelf

build_fullelf_ctrlpkt: ${elf_ctrlpkt} test_fullelf

build_xclbin: ${xclbin} ${insts_bin} test_xclbin

build_runlist: ${xclbin} ${insts_bin} test_runlist

run_fullelf_loadpdi: build_fullelf_loadpdi
	./test_fullelf ${elf_loadpdi} ${CORES}

run_fullelf_blockwrites: build_fullelf_blockwrites
	./test_fullelf ${elf_blockwrites} ${CORES}

run_fullelf_ctrlpkt: build_fullelf_ctrlpkt
	./test_fullelf ${elf_ctrlpkt} ${CORES}

run_xclbin: build_xclbin
	./test_xclbin ${xclbin} ${insts_bin} ${CORES}

run_runlist: build_runlist
	./test_runlist ${xclbin} ${insts_bin} ${CORES} ${RECONFIGS}

clean:
	rm -rf build test_fullelf test_xclbin test_runlist

# ---
# AIE Array design
# ---

# --- full ELF ---

build/reconfig_${suffix}.mlir: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow reconfig --cores ${CORES} --nops ${NOPS} --reconfigs ${RECONFIGS} > build/reconfig_${suffix}.mlir

${elf_loadpdi}: build/reconfig_${suffix}.mlir
	mkdir -p build
	cd build && aiecc -v --get-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos ${<F}
	mv build/aie.elf $@

${elf_blockwrites}: build/reconfig_${suffix}.mlir
	mkdir -p build
	cd build && aiecc -v --get-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos --expand-load-pdis ${<F}
	mv build/aie.elf $@

${elf_ctrlpkt}: build/reconfig_${suffix}.mlir
	mkdir -p build
	cd build && aiecc -v --get-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos --load-pdi-to-ctrlpkt ${<F}
	mv build/aie.elf $@


# ---
# Host CPU executable
# ---

XRT_FLAGS := -I${XILINX_XRT}/include -L${XILINX_XRT}/lib -luuid -lxrt_coreutil -lrt -lstdc++

test_fullelf: ${srcdir}/test.cpp
	${CXX} -std=c++17 -DFULL_ELF $< -o $@ ${XRT_FLAGS}

test_xclbin: ${srcdir}/test.cpp
	${CXX} -std=c++17 $< -o $@ ${XRT_FLAGS}

test_runlist: ${srcdir}/test.cpp
	${CXX} -std=c++17 -DRUNLIST $< -o $@ ${XRT_FLAGS}


# ---- single flow: one device (no load_pdi), xclbin + insts ----
${xclbin} ${insts_bin} &: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow single --cores ${CORES} --nops ${NOPS} > build/single_${suffix}.mlir
	cd build && aiecc --no-xchesscc --no-xbridge --get-xclbin --get-npu-insts \
		--xclbin-name=$(notdir ${xclbin}) --npu-insts-name=$(notdir ${insts_bin}) \
		single_${suffix}.mlir

