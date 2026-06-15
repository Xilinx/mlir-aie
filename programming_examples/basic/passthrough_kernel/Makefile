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
in1_size = 4096 # in bytes (also element count, since the kernel is uint8)
data_size = ${in1_size} # transfer size used to name dynamic-TXN build artifacts
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
# Generate parameterized C++ TXN from dynamic Python design.
# The generated function takes buffer_length as a runtime parameter.
# Uses aiecc to run the full pipeline (placement, objectFIFO lowering, etc.)
# and generate both XCLBIN and C++ TXN in one invocation.
build/dynsize/aie_dynamic.mlir: ${srcdir}/passthrough_kernel_dynamic.py
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} > $@

build/dynsize/final_dynamic.xclbin build/dynsize/generated_txn.h &: build/dynsize/aie_dynamic.mlir build/passThrough.cc.o
	mkdir -p build/dynsize
	cp build/passThrough.cc.o build/dynsize/
	cd build/dynsize && aiecc --aie-generate-xclbin --xclbin-name=final_dynamic.xclbin \
		--aie-generate-txn-cpp --txn-cpp-name=generated_txn.h \
		--no-xchesscc --no-xbridge \
		$(<:%=../../%)

# Build the dynamic test executable (reuses test.cpp with USE_DYNAMIC_TXN)
${targetname}_dynamic_${data_size}.exe: ${srcdir}/test.cpp build/dynsize/generated_txn.h
	rm -rf _build_dynamic
	mkdir -p _build_dynamic
	cd _build_dynamic && ${powershell} cmake `${getwslpath} ${srcdir}` \
		-DTARGET_NAME=${targetname}_dynamic \
		-DIN1_SIZE=${in1_size} -DOUT_SIZE=${in1_size} \
		-DCMAKE_CXX_FLAGS="-DUSE_DYNAMIC_TXN -I${CURDIR}/build/dynsize -I$(shell cd ${srcdir}/../../.. && pwd)/include"
	cd _build_dynamic && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build_dynamic/${targetname}_dynamic.exe $@
else
	cp _build_dynamic/${targetname}_dynamic $@
endif

# Build dynamic-size executable using parameterized TXN (takes --dynamic-size at runtime)
max_size = 4096
${targetname}_dynsize.exe: ${srcdir}/test.cpp build/dynsize/generated_txn.h
	rm -rf _build_dynsize
	mkdir -p _build_dynsize
	cd _build_dynsize && ${powershell} cmake `${getwslpath} ${srcdir}` \
		-DTARGET_NAME=${targetname}_dynsize \
		-DIN1_SIZE=${max_size} -DOUT_SIZE=${max_size} \
		-DCMAKE_CXX_FLAGS="-DUSE_DYNAMIC_TXN -I${CURDIR}/build/dynsize -I$(shell cd ${srcdir}/../../.. && pwd)/include"
	cd _build_dynsize && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build_dynsize/${targetname}_dynsize.exe $@
else
	cp _build_dynsize/${targetname}_dynsize $@
endif

# Run with dynamically generated TXN and its own XCLBIN
run_dynamic: ${targetname}_dynamic_${data_size}.exe build/dynsize/final_dynamic.xclbin
	${powershell} ./$< -x build/dynsize/final_dynamic.xclbin -k MLIR_AIE

# Build XCLBIN at max size, then run at multiple sizes without recompilation
run_dynamic_sizes: ${targetname}_dynsize.exe build/final_${data_size}.xclbin
	@for sz in 1024 4096 16384 ${in1_size}; do \
		echo "=== Dynamic size: $$sz bytes ==="; \
		${powershell} ./$< -x build/final_${data_size}.xclbin -k MLIR_AIE --dynamic-size $$sz; \
		echo ""; \
	done

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build _build_dynamic _build_dynsize ${targetname}*.exe
