##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Softmax Example Makefile
#
# USAGE:
#   make [target] [VARIABLE=value ...]
#
# EXAMPLES:
#   make                                    # Build xclbin (default)
#   make run                                # Build and run with defaults
#   make run use_whole_array=1 whole_array_cols=4 whole_array_rows=4
#   make run size=524288 n_iterations=100   # Custom size and iterations
#   make trace                              # Run with hardware tracing
#   make clean                              # Remove build artifacts
#   make help                               # Show this help message
#
# NOTE: Configuration changes (cols, rows, size, etc.) are automatically
#       detected - no need to run 'make clean' when changing parameters.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

#------------------------------------------------------------------------------
# Device Configuration
#------------------------------------------------------------------------------

# Target device: npu (AIE2) or npu2 (AIE2P)
devicename ?= npu

# Runtime library path (for npu/AIE2 only)
aie2_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2

# Kernel source path based on device
ifeq ($(devicename),npu2)
VPATH := ${srcdir}/../../../aie_kernels/aie2p
else
VPATH := ${srcdir}/../../../aie_kernels/aie2
endif

#------------------------------------------------------------------------------
# Runtime Parameters
#------------------------------------------------------------------------------

# Input data size (number of elements)
size ?= 262144

# Benchmark iterations
n_iterations ?= 20
n_warmup ?= 10

# Trace buffer size (bytes)
trace_size = 8192

#------------------------------------------------------------------------------
# Placement Options
#------------------------------------------------------------------------------

# Placement mode (mutually exclusive):
#   use_placed=0, use_whole_array=0  -> Default (softmax.py)
#   use_placed=1                     -> Manual placement (softmax_placed.py)
#   use_whole_array=1                -> Whole array placement (softmax_whole_array_placed.py)
use_placed ?= 0
use_whole_array ?= 0

# Whole array configuration (only used when use_whole_array=1)
whole_array_cols ?= 1
whole_array_rows ?= 4

#------------------------------------------------------------------------------
# Internal Variables (do not modify)
#------------------------------------------------------------------------------

targetname = softmax

# Select Python source and arguments based on placement mode
aie_py_src = ${targetname}.py
aie_py_args = --size ${size}

ifeq (${use_placed}, 1)
aie_py_src = ${targetname}_placed.py
endif

ifeq (${use_whole_array}, 1)
aie_py_src = ${targetname}_whole_array_placed.py
aie_py_args = --n_col ${whole_array_cols} --n_cores_per_col ${whole_array_rows} --size ${size}
endif

# Config string to track build-affecting variables
# When any of these change, dependent targets will be rebuilt automatically
build_config := $(use_placed)-$(use_whole_array)-$(whole_array_cols)-$(whole_array_rows)-$(size)-$(devicename)

#------------------------------------------------------------------------------
# Main Targets
#------------------------------------------------------------------------------

.PHONY: all run profile trace clean help FORCE

# Default target: build xclbin and instructions
all: build/final.xclbin build/insts.bin

# Run the softmax example
ifeq ($(devicename),npu)
run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE \
		--npu 1 --iters ${n_iterations} --warmup ${n_warmup} --size ${size}
else ifeq ($(devicename),npu2)
run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE \
		--npu 2 --iters ${n_iterations} --warmup ${n_warmup} --size ${size}
endif

# Run with profiling (outputs to results.csv)
profile: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -p results.csv

# Run with hardware tracing enabled
trace: ${targetname}.exe build/final_trace.xclbin
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir build/aie_trace.mlir --output trace_softmax.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_softmax.json

# Remove all build artifacts
clean:
	rm -rf build _build ${targetname}.exe

# Display help message
help:
	@echo "Softmax Example Makefile"
	@echo ""
	@echo "TARGETS:"
	@echo "  all       Build xclbin and instructions (default)"
	@echo "  run       Build and run the softmax example"
	@echo "  profile   Run with profiling (outputs to results.csv)"
	@echo "  trace     Run with hardware tracing enabled"
	@echo "  clean     Remove all build artifacts"
	@echo "  help      Show this help message"
	@echo ""
	@echo "CONFIGURATION VARIABLES:"
	@echo "  devicename        Target device: npu or npu2 (default: npu)"
	@echo "  size              Input data size (default: 262144)"
	@echo "  n_iterations      Benchmark iterations (default: 20)"
	@echo "  n_warmup          Warmup iterations (default: 10)"
	@echo ""
	@echo "PLACEMENT OPTIONS:"
	@echo "  use_placed        Use manual placement mode (default: 0)"
	@echo "  use_whole_array   Use whole array placement (default: 0)"
	@echo "  whole_array_cols  Number of columns (default: 1)"
	@echo "  whole_array_rows  Cores per column (default: 4)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make run"
	@echo "  make run use_whole_array=1 whole_array_cols=4 whole_array_rows=4"
	@echo "  make run size=524288 n_iterations=100"
	@echo "  make trace"
	@echo ""
	@echo "NOTE: Configuration changes are automatically detected."
	@echo "      No need to run 'make clean' when changing parameters."

#------------------------------------------------------------------------------
# Kernel Compilation
#------------------------------------------------------------------------------

# Compile softmax kernel to object file
build/softmax.o: ${VPATH}/softmax.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
		-I. -I${aie2_runtime_dir} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	@echo "Device type not supported: $(devicename)"
	@exit 1
endif

# LUT-based operations (npu/AIE2 only)
ifeq ($(devicename),npu)
build/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}
endif

# Create kernel library
ifeq ($(devicename),npu)
build/kernels.a: build/softmax.o build/lut_based_ops.o
	ar rvs $@ $+
else ifeq ($(devicename),npu2)
build/kernels.a: build/softmax.o
	ar rvs $@ $+
endif

#------------------------------------------------------------------------------
# Debug Targets (for kernel inspection)
#------------------------------------------------------------------------------

# Generate assembly code for the softmax kernel
build/softmax.s: ${VPATH}/softmax.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
		-I. -I${aie2_runtime_dir} -S $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -S $< -o ${@F}
else
	@echo "Device type not supported: $(devicename)"
	@exit 1
endif

# Generate LLVM IR for the softmax kernel
build/softmax.ll: ${VPATH}/softmax.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} \
		-I. -I${aie2_runtime_dir} -S -emit-llvm $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -S -emit-llvm $< -o ${@F}
else
	@echo "Device type not supported: $(devicename)"
	@exit 1
endif

#------------------------------------------------------------------------------
# MLIR Generation
#------------------------------------------------------------------------------

# Config stamp: triggers rebuild when configuration changes
build/.config_stamp: FORCE
	@mkdir -p ${@D}
	@echo '$(build_config)' | cmp -s - $@ || echo '$(build_config)' > $@

FORCE:

# Generate AIE MLIR from Python
build/aie.mlir: ${srcdir}/${aie_py_src} build/.config_stamp
	mkdir -p ${@D}
	python3 $< ${devicename} ${aie_py_args} > $@

# Generate AIE MLIR with tracing enabled
build/aie_trace.mlir: ${srcdir}/${aie_py_src} build/.config_stamp
	mkdir -p ${@D}
	python3 $< ${devicename} --trace_size ${trace_size} ${aie_py_args} > $@

#------------------------------------------------------------------------------
# XCLBIN Generation
#------------------------------------------------------------------------------

# Build final xclbin
build/final.xclbin: build/aie.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

# Build xclbin with tracing enabled
build/final_trace.xclbin: build/aie_trace.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

#------------------------------------------------------------------------------
# Host Executable
#------------------------------------------------------------------------------

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif
