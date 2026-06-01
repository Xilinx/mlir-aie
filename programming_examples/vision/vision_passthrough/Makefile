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

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
PASSTHROUGH_WIDTH = 1920
PASSTHROUGH_HEIGHT = 1080

#PASSTHROUGH_WIDTH = 512
#PASSTHROUGH_HEIGHT = 9

targetname = vision_passthrough

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -W ${PASSTHROUGH_WIDTH} -H ${PASSTHROUGH_HEIGHT}

xclbin_target = build/final_${PASSTHROUGH_WIDTH}.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp,-DPASSTHROUGH_WIDTH=${PASSTHROUGH_WIDTH} -DPASSTHROUGH_HEIGHT=${PASSTHROUGH_HEIGHT} -D CMAKE_C_COMPILER=${CC} -D CMAKE_CXX_COMPILER=${CXX}))

run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe ${targetname}Out_test.jpg
