##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = mlir_trace_example

tensor_elems = 4096
tile_size    = 1024
# in1_size / out_size are in bytes (int32 == 4); kept in sync with tensor_elems.
in1_size     = 16384
in2_size     = 4
out_size     = 16384
trace_size   = 8192

aie_py_src = aie_trace.py
aieargs = -d ${devicename} --tensor-size ${tensor_elems} --tile-size ${tile_size} --trace-size ${trace_size}

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin

.PHONY: all
all: run_trace_py

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp,-DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}))

.PHONY: run_trace
run_trace: ${targetname}.exe ${xclbin_target} ${insts_target}
	@echo "Running C++ test with custom-event trace..."
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir $(xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output trace.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace.json -o trace_timeline.png -t "Declarative Trace"
	@echo "Trace visualization saved to trace_timeline.png"

.PHONY: run_trace_py
run_trace_py: ${xclbin_target} ${insts_target}
	@echo "Running Python test with custom-event trace..."
	python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} --kernel MLIR_AIE --verbosity 1 --trace_size ${trace_size} --trace-file trace.txt
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir $(xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output trace.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace.json
	@echo ""
	@echo "Generating trace visualization..."
	python3 ${srcdir}/visualize_trace.py -i trace.json -o trace_timeline.png -t "Declarative Trace (Python)"
	@echo "Trace visualization saved to trace_timeline.png"

.PHONY: clean_trace
clean_trace:
	rm -rf tmpTrace trace.txt trace_timeline.png trace.json

.PHONY: clean
clean: clean_trace
	rm -rf build _build ${targetname}*.exe
