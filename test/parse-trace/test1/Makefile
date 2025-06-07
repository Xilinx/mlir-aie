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

all:
	${srcdir}/../../utils/parse_trace.py --input trace_test1.txt --mlir aie_test1.mlir --output trace_test1.json
	${srcdir}/../../utils/get_trace_summary.py --input trace_test1.json


clean: 
	rm -rf *json
