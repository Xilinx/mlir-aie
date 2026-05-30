##===- Makefile -----------------------------------------------------------===##
#
# Phase 1 dataflow sanity build for llama_3_2_1b.
# Builds the stub gemm_int8_srs frame (1 CT, 4-in/1-out ObjectFifos, stub
# kernel = act -> out passthrough) and runs it on the NPU.
#
# Usage (with the mlir-aie env active):
#   make gemm_pt        -> build the stub xclbin
#   make run_gemm_pt    -> build + run on NPU, bit-exact passthrough check
#   make clean
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

BUILD := ${srcdir}/build
KDIR  := ${srcdir}/kernels

# Pinned tiny shape for the stub.
M ?= 8
K ?= 64
N ?= 64

PEANO_FLAGS := ${PEANOWRAP2P_FLAGS} \
    -DLLAMA_GEMM_PT_M=${M} -DLLAMA_GEMM_PT_K=${K} -DLLAMA_GEMM_PT_N=${N}

.PHONY: all clean gemm_pt run_gemm_pt

all: gemm_pt

# ---------- kernel object ----------
${BUILD}/llama_gemm_int8_srs_pt.cc.o: ${KDIR}/llama_gemm_int8_srs_pt.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANO_FLAGS} -c $< -o ${@F}

# ---------- MLIR + xclbin ----------
${BUILD}/aie_gemm_pt_${M}x${K}x${N}.mlir: ${srcdir}/aie2_gemm_int8_srs.py
	mkdir -p ${@D}
	python3 $< -M ${M} -K ${K} -N ${N} > $@

${BUILD}/final_gemm_pt_${M}x${K}x${N}.xclbin: ${BUILD}/aie_gemm_pt_${M}x${K}x${N}.mlir \
                                              ${BUILD}/llama_gemm_int8_srs_pt.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_gemm_pt_${M}x${K}x${N}.bin \
		$(notdir $<)

gemm_pt: ${BUILD}/final_gemm_pt_${M}x${K}x${N}.xclbin

# ---------- run ----------
run_gemm_pt: gemm_pt
	cd ${srcdir} && python3 test_gemm_int8_srs_pt.py \
		-x ${BUILD}/final_gemm_pt_${M}x${K}x${N}.xclbin \
		-i ${BUILD}/insts_gemm_pt_${M}x${K}x${N}.bin \
		-k MLIR_AIE -M ${M} -K ${K} -N ${N}

# ---------- clean ----------
clean:
	rm -rf ${BUILD}
