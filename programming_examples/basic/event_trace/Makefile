##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

all: run_trace_py

build/insts.bin: build/final.xclbin

# Compile the AIE kernel
build/scale.o: ${srcdir}/vector_scalar_mul.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

# Preprocess the MLIR: substitute NPUDEVICE with the target device string
build/aie_trace.mlir: ${srcdir}/aie_trace.mlir
	mkdir -p ${@D}
	sed 's/NPUDEVICE/${devicename}_1col/g' $< > $@

# Generate MLIR from Python
build/aie_trace_from_py.mlir: ${srcdir}/aie_trace.py
	mkdir -p ${@D}
	python3 $< > $@

# Build xclbin from the mlir
build/final.xclbin: build/aie_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			--no-xchesscc --no-xbridge --dump-intermediates \
			--aie-generate-npu-insts --npu-insts-name=insts.bin aie_trace.mlir

# Build xclbin from Python-generated mlir
build/final_py.xclbin: build/aie_trace_from_py.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			--no-xchesscc --no-xbridge --dump-intermediates \
			--aie-generate-npu-insts --npu-insts-name=insts_py.bin aie_trace_from_py.mlir

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
run_trace: ${targetname}.exe build/final.xclbin build/insts.bin
	@echo "Running with declarative trace syntax..."
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/aie_trace.mlir.prj/main_physical_with_elfs.mlir --output trace.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace.json

	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace.json -o trace_timeline.png -t "Declarative Trace"
	@echo "Trace visualization saved to trace_timeline.png"

# Run Python version with new trace syntax
run_trace_py: build/final.xclbin build/insts.bin
	@echo "Running Python test with declarative trace syntax..."
	python3 ${srcdir}/test.py --xclbin build/final.xclbin --instr build/insts.bin --kernel MLIR_AIE --verbosity 1 --trace-sz ${trace_size} --trace-file trace.txt
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/aie_trace.mlir.prj/main_physical_with_elfs.mlir --output trace.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace.json -o trace_timeline.png -t "Declarative Trace (Python)"
	@echo "Trace visualization saved to trace_timeline.png"

# Generate MLIR from Python and build/run
run_trace_from_py: ${targetname}.exe build/final_py.xclbin
	@echo "Running with Python-generated declarative trace MLIR..."
	${powershell} ./$< -x build/final_py.xclbin -i build/insts_py.bin -k MLIR_AIE -t ${trace_size}

# Just generate and print the MLIR from the Python bindings
generate_mlir_from_py: build/aie_trace_from_py.mlir
	@cat $<

clean_trace:
	rm -rf tmpTrace trace.txt trace_timeline.png trace.json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe

.PHONY: all run_trace run_trace_py run_trace_from_py generate_mlir_from_py clean clean_trace
