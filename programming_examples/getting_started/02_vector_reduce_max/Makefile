##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Set SRC_DIR to the directory in which this Makefile sits.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

# Configuration of this example.
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_reduce_max_1col
aie_py_src = ${targetname}.py

# ----------------------------------------------------------------------------
# Build Targets
# ----------------------------------------------------------------------------

.PHONY: all template clean

all: build/final.xclbin

# External single compute core kernel code defined in C source
build/vector_reduce_max.o: ${srcdir}/vector_reduce_max.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

# Lower-level MLIR design built from higher-level Python IRON design
build/aie2.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

# The final.xclbin contains static configuration for the NPU (including data 
# movement configuration, compute core program memories).
# insts.bin contains instructions for the command processor to execute
# at runtime (i.e., dispatch running the design, read from DRAM, etc.).
build/final.xclbin: build/aie2.mlir build/vector_reduce_max.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

# Host code: Program executed on the CPU that will invoke our NPU design
# generated below and verify its outputs.
${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DIN1_SIZE=524288 -DOUT_SIZE=4 -DDTYPE=std::bfloat16_t
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE 

clean:
	rm -rf build _build ${targetname}*.exe