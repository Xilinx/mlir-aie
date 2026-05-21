##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

device ?= $(if $(filter 1,$(NPU2)),npu2,npu)
PASSTHROUGH_WIDTH = 1920
PASSTHROUGH_HEIGHT = 1080

#PASSTHROUGH_WIDTH = 512
#PASSTHROUGH_HEIGHT = 9

targetname = vision_passthrough

aie_py_src = ${targetname}.py
aieargs = -d ${device} -W ${PASSTHROUGH_WIDTH} -H ${PASSTHROUGH_HEIGHT}

xclbin_target = build/final_${PASSTHROUGH_WIDTH}.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

${xclbin_target} ${insts_target} &: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target}

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DPASSTHROUGH_WIDTH=${PASSTHROUGH_WIDTH} -DPASSTHROUGH_HEIGHT=${PASSTHROUGH_HEIGHT} -D CMAKE_C_COMPILER=${CC} -D CMAKE_CXX_COMPILER=${CXX}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe ${targetname}Out_test.jpg
