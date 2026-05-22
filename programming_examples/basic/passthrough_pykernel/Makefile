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
# Drives the unified @iron.jit passthrough_pykernel.py via --xclbin-path /
# --insts-path; test.cpp + test.py consume the artifacts.  The standalone
# notebook (passthrough_pykernel.ipynb) is independent of this Makefile's
# build flow.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = passthrough_pykernel
data_size = 4096

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -s ${data_size}

xclbin_target = build/final_${data_size}.xclbin
insts_target = build/insts_${data_size}.bin

.PHONY: all template clean clean_notebook

all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DPASSTHROUGH_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}_${data_size}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py -s ${data_size} -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}*.exe

run_notebook: ${srcdir}/passthrough_pykernel.ipynb
	mkdir -p notebook_build
	cd notebook_build && jupyter nbconvert --to script ${srcdir}/passthrough_pykernel.ipynb --output-dir .
	cd notebook_build && ipython passthrough_pykernel.py

clean_notebook:
	rm -rf notebook_build notebook_aie.mlir
