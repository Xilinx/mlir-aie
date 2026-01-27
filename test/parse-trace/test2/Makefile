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

all:
	${srcdir}/../../../python/utils/trace/parse.py --input ${srcdir}/trace_test2.txt --mlir ${srcdir}/aie_test2.mlir --output trace_test2.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_test2.json

diff:
	diff trace_test2.json ${srcdir}/golden_json.txt

clean: 
	rm -rf trace_test2.json
