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

# Visualize existing trace data
visualize: trace_mlir.json
	python3 ${srcdir}/visualize_trace.py -i trace_mlir.json -o trace_timeline.png
	@echo "Trace visualization saved to trace_timeline.png"

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json trace_timeline.png

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all run trace visualize clean clean_trace
