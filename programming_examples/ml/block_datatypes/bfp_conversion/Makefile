##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

targetname = bfp_conversion
use_chess?=0
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
col = 0

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final.xclbin

build/kernel.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
ifeq ($(use_chess),1)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
endif
else
	echo "Device type not supported in this example"
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${col} > $@

build/final.xclbin: build/aie.mlir build/kernel.o
	mkdir -p ${@D}
ifeq ($(use_chess),1)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)
endif

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< --verbosity=2 -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
