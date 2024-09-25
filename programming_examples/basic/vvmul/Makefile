##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

VPATH := ${srcdir}/../../../aie_kernels/aie2

data_size = 4096
targetname = vectorMult
devicename = npu
col = 0

all: build/final.xclbin

build2/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} ${data_size} > $@

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}  -DDATASIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif


build2/aie.mlir: ../vector_scalar_mul/aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} 0 > $@

build2/final.xclbin: build2/aie.mlir build2/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)


# Changing variables when we target VCK5000
vck5000: devicename=xcvc1902
vck5000: col=6

vck5000: build/aie.mlir
	aiecc.py --xchesscc --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie.mlir \
		-I/opt/xaiengine/include \
		-I${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/include \
		-L/opt/xaiengine/lib \
		-L/lib/x86_64-linux-gnu/ \
		${srcdir}/test_vck5000.cpp \
		${srcdir}/../../../install/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		-Wl,-R/opt/xaiengine/lib \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o test.elf


run: ${targetname}.exe build/final.xclbin build/insts.txt build2/final.xclbin
	${powershell} ./$< --xclbin build/final.xclbin --xclbin_2 build2/final.xclbin --instr build/insts.txt --instr_2 build2/insts.txt -k MLIR_AIE

clean:
	rm -rf build build2 _build inst aie.mlir.prj core_* test.elf ${targetname}.exe build2/aie.mlir build2/final.xclbin
