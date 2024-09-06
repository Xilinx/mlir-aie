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

data_size = 4096
targetname = vectorMult
devicename = npu
col = 0

all: build/final.xclbin

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} ${data_size} > $@

build/final.xclbin: build/aie.mlir ../vector_scalar_mul/build/final.xclbin
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --xclbin-kernel-name=VVM --xclbin-kernel-id=0x902 \
				--xclbin-instance-name=VVMINST --no-aiesim  --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--xclbin-input=../../vector_scalar_mul/build/final.xclbin  \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

../vector_scalar_mul/build/aie.mlir: ../vector_scalar_mul/aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} 0 > $@

../vector_scalar_mul/build/final.xclbin: ../vector_scalar_mul/build/aie.mlir
	cd ../vector_scalar_mul/build/ && aiecc.py --xclbin-kernel-name=VSM --xclbin-kernel-id=0x901 \
			--xclbin-instance-name=VSMINST --no-aiesim  --aie-generate-cdo --no-compile-host --xclbin-name=final.xclbin \
			--aie-generate-npu --npu-insts-name=insts.txt aie.mlir


${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DDATASIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif



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


run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< --xclbin build/final.xclbin \
	--instr build/insts.txt --instr_2 ../vector_scalar_mul/build/insts.txt -k VVM

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe ../vector_scalar_mul/build/final.xclbin ../vector_scalar_mul/aie.mlir
