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
VPATH := ${srcdir}/../../../aie_kernels/generic

# Configuration of this example.
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = memcpy
length ?= 16384
cols ?= 1
chans ?= 1
bypass ?= True

aie_py_src=memcpy.py

# ----------------------------------------------------------------------------
# Build Targets
# ----------------------------------------------------------------------------

.PHONY all: build/${targetname}.mlir build/final.xclbin

# External single compute core kernel code defined in C source
build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

# Lower-level MLIR design built from higher-level Python IRON design
build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -l ${length} -co ${cols} -ch ${chans} -b ${bypass} > $@

# The final.xclbin contains static configuration for the NPU (including data 
# movement configuration, compute core program memories). 
build/final.xclbin: build/${targetname}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host \
	            --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-elf --elf-name=insts.elf $(<:%=../%)

# Host code: Program executed on the CPU that will invoke our NPU design
# and verify its outputs.
${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.elf -k MLIR_AIE -l ${length}

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
