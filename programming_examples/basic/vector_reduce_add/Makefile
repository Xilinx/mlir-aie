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

targetname = vector_reduce_add

aie_py_src = ${targetname}.py
aieargs = -d ${devicename}

xclbin_target = build/final.xclbin
insts_target = build/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

trace:
	../../../python/utils/trace/parse.py --input trace.txt --mlir $(xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt

clean: clean_trace
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe
