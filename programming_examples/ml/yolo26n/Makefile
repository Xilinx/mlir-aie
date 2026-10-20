##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Ultralytics yolo26n detection on NPU2: fetching and quantizing the
# checkpoint (export), extracting its integer parameters (data; synthetic
# ones are made under BUILD as runs need them), checking the oracle against ONNX
# Runtime (verify_oracle), running any operator range on the NPU against the
# oracle (run_block, run_blocks_synthetic) and the whole model on images (run,
# run_synthetic). README.md has the usual invocations.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

BUILD ?= ${srcdir}/build
SEED ?= 7
N ?= 8

.PHONY: all clean export data verify_oracle \
	run_block run_blocks_synthetic run run_synthetic

all: run_blocks_synthetic

export:
	cd ${srcdir} && python3 export_yolo26n.py

data:
	cd ${srcdir} && python3 gen_yolo_data.py

SYNTHETIC_PARAMS := ${BUILD}/yolo26n_params_synthetic${SEED}.npz

verify_oracle:
	cd ${srcdir} && python3 verify_oracle.py --n ${N} $(if ${IMAGES},--images ${IMAGES})

# One operator range on the real weights, e.g. BLOCK=52:59; OUTPUTS names the
# tensors to drain when the range ends in more than one, SYNTHETIC=SEED swaps
# in seeded weights.
run_block:
	$(if ${BLOCK},,$(error set BLOCK=FIRST:STOP))
	cd ${srcdir} && python3 bringup.py --ops ${BLOCK} \
		$(if ${OUTPUTS},--outputs ${OUTPUTS}) \
		$(if ${SYNTHETIC},--synthetic ${SYNTHETIC} --params ${BUILD}/yolo26n_params_synthetic${SYNTHETIC}.npz)

# One range of each block kind: the stem and C3k2, SPPF, C2PSA attention, an
# upsample/concat neck stage, one Detect level with both branches, and the
# postprocess. FIRST:STOP=T1,T2 drains T1 and T2 instead of the last op's output.
DETECT0 := /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv,/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv
SYNTHETIC_BLOCKS := 0:11 52:59 59:73 90:107 143:151=${DETECT0} 167:168

run_blocks_synthetic:
	@cd ${srcdir} && passed=0 && total=0 && \
	for block in ${SYNTHETIC_BLOCKS}; do \
		case $$block in \
			*=*) set -- --outputs "$${block#*=}" ;; \
			*) set -- ;; \
		esac; \
		echo "--- ops $${block%%=*}"; \
		if python3 bringup.py --ops "$${block%%=*}" --synthetic ${SEED} --params ${SYNTHETIC_PARAMS} "$$@"; then \
			passed=$$((passed + 1)); \
		fi; \
		total=$$((total + 1)); \
	done; \
	echo "yolo26n blocks: $$passed/$$total PASS"; \
	test $$passed -eq $$total

run:
	cd ${srcdir} && python3 yolo26n.py --n ${N} $(if ${IMAGES},--images ${IMAGES})

run_synthetic:
	cd ${srcdir} && python3 yolo26n.py --n ${N} --synthetic ${SEED} --params ${SYNTHETIC_PARAMS}

clean:
	rm -rf ${BUILD}
