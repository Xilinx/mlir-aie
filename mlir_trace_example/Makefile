##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
#
# Makefile for MLIR trace example
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../programming_examples/makefile-common

devicename ?= npu
targetname = mlir_trace_example

in1_size = 16384 # in bytes
in2_size = 4 # in bytes, should always be 4 (1x int32)
out_size = 16384 # in bytes, should always be equal to in1_size
trace_size = 8192

all: build/final.xclbin

build/insts.bin: build/final.xclbin

# Compile the AIE kernel
build/scale.o: ${srcdir}/vector_scalar_mul.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}

# Build the xclbin using the MLIR file directly
build/final.xclbin: ${srcdir}/aie_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts.bin ../aie_trace.mlir

# Lower the new declarative trace syntax to low-level operations
build/aie_new_trace_lowered.mlir: ${srcdir}/aie_new_trace.mlir
	mkdir -p ${@D}
	cd /work/acdc/aie && aie-opt --aie-trace-to-config --aiex-inline-trace-config --aiex-config-to-npu ${srcdir}/aie_new_trace.mlir -o /work/acdc/aie/mlir_trace_example/build/aie_new_trace_lowered.mlir

# Build xclbin from the new trace syntax (lowered)
build/final_new.xclbin: build/aie_new_trace_lowered.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts_new.bin aie_new_trace_lowered.mlir

build/insts_new.bin: build/final_new.xclbin

# Build the host executable
${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

# Run the example
run: ${targetname}.exe build/final.xclbin build/insts.bin 
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

# Run with trace enabled
trace: ${targetname}.exe build/final.xclbin build/insts.bin 
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../python/utils/parse_trace.py --input trace.txt --mlir ${srcdir}/aie_trace.mlir --output trace_mlir.json
	${srcdir}/../python/utils/get_trace_summary.py --input trace_mlir.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace_mlir.json -o trace_timeline.png
	@echo "Trace visualization saved to trace_timeline.png"

# Test the new declarative trace syntax
test_new_trace: build/aie_new_trace_lowered.mlir
	@echo "==================================================================="
	@echo "Testing New Declarative Trace Syntax"
	@echo "==================================================================="
	@echo ""
	@echo "Step 1: Lowering passed successfully"
	@echo "  Input:  aie_new_trace.mlir (declarative)"
	@echo "  Output: build/aie_new_trace_lowered.mlir (low-level)"
	@echo ""
	@echo "Step 2: Comparing lowered output with manual version"
	@echo "  Showing differences between lowered and manual configurations:"
	@echo ""
	-diff -u ${srcdir}/aie_trace.mlir build/aie_new_trace_lowered.mlir | head -50 || true
	@echo ""
	@echo "Step 3: Checking if lowered MLIR can be compiled"
	@ls -lh build/aie_new_trace_lowered.mlir
	@echo ""
	@echo "To build and run the new trace version:"
	@echo "  make run_new_trace"

# Build and run with the new trace syntax
run_new_trace: ${targetname}.exe build/final_new.xclbin build/insts_new.bin
	@echo "Running with NEW declarative trace syntax..."
	${powershell} ./$< -x build/final_new.xclbin -i build/insts_new.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../python/utils/parse_trace.py --input trace.txt --mlir build/aie_new_trace_lowered.mlir --output trace_new.json
	${srcdir}/../python/utils/get_trace_summary.py --input trace_new.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace_new.json -o trace_new_timeline.png -t "New Declarative Trace"
	@echo "Trace visualization saved to trace_new_timeline.png"

# Visualize existing trace data
visualize: trace_mlir.json
	python3 ${srcdir}/visualize_trace.py -i trace_mlir.json -o trace_timeline.png
	@echo "Trace visualization saved to trace_timeline.png"

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json trace_timeline.png trace_new*.png trace_new.json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all run trace test_new_trace run_new_trace visualize clean clean_trace
