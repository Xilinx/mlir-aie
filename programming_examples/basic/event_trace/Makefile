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

include ${srcdir}/../../makefile-common

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

# Lower the new declarative trace syntax to low-level operations
build/aie_new_trace_lowered.mlir: ${srcdir}/aie_new_trace.mlir
	mkdir -p ${@D}
	cd /work/acdc/aie && aie-opt --aie-trace-to-config --aie-inline-trace-config --aiex-config-to-npu ${srcdir}/aie_new_trace.mlir -o /work/acdc/aie/mlir_trace_example/build/aie_new_trace_lowered.mlir

# Build xclbin from the mlir
build/final.xclbin: ${srcdir}/aie_new_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts.bin ${srcdir}/aie_new_trace.mlir

build/insts.bin: build/final.xclbin

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

# Build and run with the new trace syntax
run_new_trace: ${targetname}.exe build/final.xclbin build/insts.bin
	@echo "Running with NEW declarative trace syntax..."
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/aie_new_trace.mlir.prj/input_physical_with_elfs.mlir --output trace_new.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_new.json

	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace_new.json -o trace_new_timeline.png -t "New Declarative Trace"
	@echo "Trace visualization saved to trace_new_timeline.png"

# Run Python version with new trace syntax
run_new_trace_py: build/final.xclbin build/insts.bin
	@echo "Running Python test with declarative trace syntax..."
	python3 ${srcdir}/test.py --xclbin build/final.xclbin --instr build/insts.bin --kernel MLIR_AIE --verbosity 1 --trace-sz ${trace_size} --trace-file trace.txt
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/aie_new_trace.mlir.prj/input_physical_with_elfs.mlir --output trace_new.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_new.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace_new.json -o trace_new_timeline.png -t "New Declarative Trace (Python)"
	@echo "Trace visualization saved to trace_new_timeline.png"

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json trace_timeline.png trace_new*.png trace_new.json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all test_new_trace run_new_trace run_new_trace_py clean clean_trace
