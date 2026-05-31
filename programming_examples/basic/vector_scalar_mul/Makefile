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
# Drives the unified @iron.jit vector_scalar_mul.py: the .py both builds
# kernels (peano or chess) and emits the xclbin/insts directly to the paths
# requested below.  Host code (test.cpp or test.py) consumes those artifacts.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = vector_scalar_mul
int_bit_width ?= 16

# Sizes are in bytes.  in2_size is always 4 (one int32 scalar factor).
ifeq (${int_bit_width}, 16)
in1_size ?= 8192
out_size ?= 8192
else  # int_bit_width == 32
in1_size ?= 16384
out_size ?= 16384
endif
in2_size = 4
trace_size ?= 8192
CHESS ?= false

ifeq (${CHESS}, true)
use_chess = 1
else
use_chess = 0
endif

data_size = $(in1_size)
aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} -bw ${int_bit_width} --use-chess ${use_chess}

xclbin_target = build/final_${data_size}.xclbin
insts_target = build/insts_${data_size}.bin
trace_xclbin_target = build/final_trace_${data_size}.xclbin
trace_insts_target = build/insts_trace_${data_size}.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call jit_xclbin,${trace_xclbin_target},${trace_insts_target},${aie_py_src},${aieargs} -t ${trace_size}))
$(eval $(call build_host_exe,${targetname}_${data_size},_build,test.cpp,-DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size} -DINT_BIT_WIDTH=${int_bit_width}))

.PHONY: run
run: ${targetname}_${data_size}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: run_py
run_py: ${xclbin_target} ${insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${xclbin_target} --instr ${insts_target} --kernel MLIR_AIE --in1-size ${in1_size} --in2-size ${in2_size} --out-size ${out_size}

# input_with_addresses.mlir is dropped into the kernel_dir (= xclbin parent) by
# @iron.jit's compile() in explicit-paths mode.
trace_mlir = build/input_with_addresses.mlir

.PHONY: trace
trace: ${targetname}_${data_size}.exe ${trace_xclbin_target} ${trace_insts_target}
	${powershell} ./$< -x ${trace_xclbin_target} -i ${trace_insts_target} -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir ${trace_mlir} --output trace_${targetname}.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_${targetname}.json

.PHONY: trace_py
trace_py: ${trace_xclbin_target} ${trace_insts_target}
	${powershell} python3 ${srcdir}/test.py --xclbin ${trace_xclbin_target} --instr ${trace_insts_target} --kernel MLIR_AIE --trace_size ${trace_size} --in1-size ${in1_size} --in2-size ${in2_size} --out-size ${out_size}
	${srcdir}/../../../python/utils/trace/parse.py --input trace.txt --mlir ${trace_mlir} --output trace_${targetname}.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_${targetname}.json

.PHONY: clean clean_trace
clean: clean_trace
	rm -rf build _build ${targetname}*.exe

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json
