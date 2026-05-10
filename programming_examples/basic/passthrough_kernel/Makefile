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

targetname = passthrough_kernel
in1_size = 4096 # in bytes (also element count, since the kernel is uint8)
trace_size = 8192

aie_py_src=${targetname}.py

.PHONY: all run run_py trace trace_py clean clean_trace

all:
	@echo "JIT mode: kernel compiles on first 'make run' / 'make run_py'."

run: run_py

run_py:
	${powershell} python3 ${srcdir}/${aie_py_src} -i1s ${in1_size}

trace: trace_py

trace_py:
	${powershell} python3 ${srcdir}/${aie_py_src} -i1s ${in1_size} -t ${trace_size}

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
