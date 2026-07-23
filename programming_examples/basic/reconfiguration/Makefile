##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Micro-benchmark of five ways to reconfigure-and-run a cols x rows core array,
# driven by a single C++ testbench (test.cpp) selected via compile-time macros:
#
#   make run_separate     separate xclbins  (worker + empty, two contexts)
#   make run_runlist      XRT runlist       (worker + empty in one runlist)
#   make run_loadpdi      full-ELF, load_pdi reconfiguration
#   make run_blockwrites  full-ELF, --expand-load-pdis (write32s + empty reset)
#   make run_ctrlpkt      full-ELF, --load-pdi-to-ctrl-pkt (control packets)
#
# Each run does ITERS timed iterations and prints per-iteration + mean/min/max
# runtimes.  Parameters COLS/ROWS/NOPS embed in the artifact names.  The
# benchmark.py script drives these targets over several array sizes.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

COLS ?= 1
ROWS ?= 1
NOPS ?= 0
RECONFIGS ?= 1
ITERS ?= 10

suffix := c${COLS}_r${ROWS}_n${NOPS}

worker_xclbin := build/worker_${suffix}.xclbin
worker_bin := build/worker_${suffix}.bin
empty_xclbin := build/empty.xclbin
empty_bin := build/empty.bin
# Runlist: WORKER + EMPTY kernels merged into one xclbin (one hw_context).
empty_k_xclbin := build/empty_k.xclbin
empty_k_bin := build/empty_k.bin
combined_xclbin := build/combined_${suffix}.xclbin
worker_rl_bin := build/worker_rl_${suffix}.bin
elf_loadpdi := build/loadpdi_${suffix}.elf
elf_blockwrites := build/blockwrites_${suffix}.elf
elf_ctrlpkt := build/ctrlpkt_${suffix}.elf

aiecc_flags := --no-xchesscc --no-xbridge --dynamic-objFifos
XRT_FLAGS := -I${XILINX_XRT}/include -L${XILINX_XRT}/lib -luuid -lxrt_coreutil -lrt -lstdc++

.PHONY: run_separate run_runlist run_loadpdi run_blockwrites run_ctrlpkt clean

# ---- designs ----

${worker_xclbin} ${worker_bin} &: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow single --cols ${COLS} --rows ${ROWS} --nops ${NOPS} > build/worker_${suffix}.mlir
	cd build && aiecc ${aiecc_flags} --get-xclbin --get-npu-insts \
		--xclbin-name=$(notdir ${worker_xclbin}) --npu-insts-name=$(notdir ${worker_bin}) worker_${suffix}.mlir

${empty_xclbin} ${empty_bin} &: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow empty > build/empty.mlir
	cd build && aiecc ${aiecc_flags} --get-xclbin --get-npu-insts \
		--xclbin-name=$(notdir ${empty_xclbin}) --npu-insts-name=$(notdir ${empty_bin}) empty.mlir

# Merge into one xclbin: build the EMPTY kernel, then the WORKER kernel with
# --xclbin-input so the result carries both PDIs (one context, two kernels).
${empty_k_xclbin} ${empty_k_bin} &: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow empty > build/empty_k.mlir
	cd build && aiecc ${aiecc_flags} --xclbin-kernel-name=EMPTY --xclbin-kernel-id=0x902 \
		--xclbin-instance-name=EMPTYINST --get-xclbin --get-npu-insts \
		--xclbin-name=$(notdir ${empty_k_xclbin}) --npu-insts-name=$(notdir ${empty_k_bin}) empty_k.mlir

${combined_xclbin} ${worker_rl_bin} &: ${srcdir}/reconfiguration.py ${empty_k_xclbin}
	mkdir -p build
	python3 $< --flow single --cols ${COLS} --rows ${ROWS} --nops ${NOPS} > build/worker_rl_${suffix}.mlir
	cd build && aiecc ${aiecc_flags} --xclbin-kernel-name=WORKER --xclbin-kernel-id=0x901 \
		--xclbin-instance-name=WORKERINST --get-xclbin --get-npu-insts --xclbin-input=$(notdir ${empty_k_xclbin}) \
		--xclbin-name=$(notdir ${combined_xclbin}) --npu-insts-name=$(notdir ${worker_rl_bin}) worker_rl_${suffix}.mlir

build/reconfig_${suffix}.mlir: ${srcdir}/reconfiguration.py
	mkdir -p build
	python3 $< --flow reconfig --cols ${COLS} --rows ${ROWS} --nops ${NOPS} --reconfigs ${RECONFIGS} > $@

${elf_loadpdi}: build/reconfig_${suffix}.mlir
	cd build && aiecc --get-full-elf --full-elf-name=$(notdir $@) ${aiecc_flags} $(notdir $<)

${elf_blockwrites}: build/reconfig_${suffix}.mlir
	cd build && aiecc --get-full-elf --expand-load-pdis --full-elf-name=$(notdir $@) ${aiecc_flags} $(notdir $<)

${elf_ctrlpkt}: build/reconfig_${suffix}.mlir
	cd build && aiecc --get-full-elf --load-pdi-to-ctrl-pkt --get-ctrlpkt --full-elf-name=$(notdir $@) ${aiecc_flags} $(notdir $<)

# ---- host testbenches (one source, three compile-time modes) ----

test_separate: ${srcdir}/test.cpp
	${CXX} -std=c++17 $< -o $@ ${XRT_FLAGS}

test_runlist: ${srcdir}/test.cpp
	${CXX} -std=c++17 -DRUNLIST $< -o $@ ${XRT_FLAGS}

test_fullelf: ${srcdir}/test.cpp
	${CXX} -std=c++17 -DFULL_ELF $< -o $@ ${XRT_FLAGS}

# ---- run targets (print runtimes; consumed by benchmark.py) ----

xclbin_args = ${worker_xclbin} ${worker_bin} ${empty_xclbin} ${empty_bin} ${COLS} ${ROWS} ${ITERS}

run_separate: ${worker_xclbin} ${worker_bin} ${empty_xclbin} ${empty_bin} test_separate
	./test_separate ${xclbin_args}

run_runlist: ${combined_xclbin} ${worker_rl_bin} ${empty_k_bin} test_runlist
	./test_runlist ${combined_xclbin} ${worker_rl_bin} ${empty_k_bin} ${COLS} ${ROWS} ${ITERS}

run_loadpdi: ${elf_loadpdi} test_fullelf
	./test_fullelf ${elf_loadpdi} ${COLS} ${ROWS} ${ITERS}

run_blockwrites: ${elf_blockwrites} test_fullelf
	./test_fullelf ${elf_blockwrites} ${COLS} ${ROWS} ${ITERS}

run_ctrlpkt: ${elf_ctrlpkt} test_fullelf
	./test_fullelf ${elf_ctrlpkt} ${COLS} ${ROWS} ${ITERS}

clean:
	rm -rf build test_separate test_runlist test_fullelf

