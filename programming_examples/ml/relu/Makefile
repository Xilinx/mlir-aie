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

include ${srcdir}/../../makefile-common

aie2_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = relu
length ?= 16384
cols ?= 4
chans ?= 2

# Input and output sizes in bytes is length * 2
in1_size = ${length} * 2
out_size = ${length} * 2

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

aie_py_src=relu.py

.PHONY all: build/${targetname}.mlir build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -I${aie2_runtime_dir} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/${targetname}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -l ${length} -co ${cols} -ch ${chans} > $@

build/final.xclbin: build/${targetname}.mlir build/${targetname}.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
	            --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=insts.bin $(<:%=../%)

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
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -l ${length}

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
