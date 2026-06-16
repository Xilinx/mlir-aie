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

targetname = passthrough_kernel
# in1_size: bytes (also element count, since the kernel is uint8).
in1_size = 4096
# data_size: transfer size used to name dynamic-TXN build artifacts.
data_size = ${in1_size}
trace_size = 8192

aie_py_src=${targetname}.py

.PHONY: all run run_py trace trace_py clean clean_trace

all:
	@echo "JIT mode: kernel compiles on first 'make run' / 'make run_py'."

run: run_py

run_py:
	${powershell} python3 ${srcdir}/${aie_py_src} -i1s ${in1_size}

trace: trace_py

trace_py:
	${powershell} python3 ${srcdir}/${aie_py_src} -i1s ${in1_size} -t ${trace_size}

# --- Dynamic TXN generation targets ---
# The dynamic design's transfer length is an SSA runtime-sequence argument, so
# the host builds the NPU instructions at runtime from the emitted C++ TXN
# header (generate_txn_sequence).  The Python driver runs the full high-level
# IRON compile: @iron.jit builds the kernel .o itself and emits the XCLBIN +
# C++ TXN header together (no static insts.bin, no hand-built .o).
build/dynsize/final_dynamic.xclbin build/dynsize/generated_txn.h &: ${srcdir}/passthrough_kernel_dynamic.py ${srcdir}/passthrough_kernel.py
	mkdir -p build/dynsize
	python3 $< -d ${devicename} -i1s ${in1_size} \
		--xclbin-path=build/dynsize/final_dynamic.xclbin

# Host-side build roots for the dynamic test.cpp (it consumes the C++ TXN
# header at runtime, so it links only against XRT — no static insts).
# MLIR_AIE_DIR (from makefile-common) is the install root whose runtime_lib
# carries the test_lib headers in the x86_64/.../include layout.
RUNTIME_LIB_ROOT ?= ${MLIR_AIE_DIR}/runtime_lib/x86_64
XRT_INC_DIR ?= /opt/xilinx/xrt/include
XRT_LIB_DIR ?= /opt/xilinx/xrt/lib
# The XRT-dependent test_utils helpers (init_xrt_load_kernel, ...) sit behind
# #ifdef TEST_UTILS_USE_XRT, which the prebuilt libtest_utils.a is not compiled
# with.  Like the cmake examples, compile test_utils.cpp from source with the
# define instead of linking the prebuilt lib.  The headers (test_utils.h,
# xrt_test_wrapper.h) come from the same repo dir so the source and its
# declarations stay in lockstep.
TEST_UTILS_DIR ?= ${srcdir}/../../../runtime_lib/test_lib
TEST_UTILS_SRC ?= ${TEST_UTILS_DIR}/test_utils.cpp

# Both dynamic executables are the same test.cpp built with USE_DYNAMIC_TXN;
# they differ only in the compiled-in default buffer size.  The generated
# header lives in build/dynsize and is included via -I.
max_size = 4096
define build_dynamic_exe
	${CXX} -std=c++23 -O2 -ggdb -DUSE_DYNAMIC_TXN -DTEST_UTILS_USE_XRT -DIN1_SIZE=$(2) -DOUT_SIZE=$(2) -o $@ $< ${TEST_UTILS_SRC} \
		-Ibuild/dynsize \
		-I${MLIR_AIE_DIR}/include \
		-I${srcdir} \
		-I${XRT_INC_DIR} -I${TEST_UTILS_DIR} \
		-I${RUNTIME_LIB_ROOT} \
		-L${XRT_LIB_DIR} -lxrt_coreutil
endef

${targetname}_dynamic_${data_size}.exe: ${srcdir}/test.cpp build/dynsize/generated_txn.h
	$(call build_dynamic_exe,$@,${in1_size})

${targetname}_dynsize.exe: ${srcdir}/test.cpp build/dynsize/generated_txn.h
	$(call build_dynamic_exe,$@,${max_size})

# Run with dynamically generated TXN and its own XCLBIN
run_dynamic: ${targetname}_dynamic_${data_size}.exe build/dynsize/final_dynamic.xclbin
	${powershell} ./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE

# One dynamic XCLBIN serves multiple sizes without recompilation: the host
# rebuilds the TXN per size from generate_txn_sequence.
run_dynamic_sizes: ${targetname}_dynsize.exe build/dynsize/final_dynamic.xclbin
	@for sz in 1024 2048 ${max_size}; do \
		echo "=== Dynamic size: $$sz bytes ==="; \
		${powershell} ./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE --dynamic-size $$sz; \
		echo ""; \
	done

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
