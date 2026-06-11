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

.PHONY: all clean gemm_pt run_gemm_pt gemm_pt_col run_gemm_pt_col gemm_pt_proj run_gemm_pt_proj flowkv_pair run_flowkv_pair glue run_glue layer_pt run_layer_pt rmsnorm run_rmsnorm rmsnorm_int8 run_rmsnorm_int8 rmsnorm_int8_dyn run_rmsnorm_int8_dyn gemm_srs run_gemm_srs silu_mul run_silu_mul rope run_rope flowkv run_flowkv sample run_sample layer_real run_layer_real chain_real run_chain_real chain_sample run_chain_sample tiled_gemm run_tiled_gemm ffn_half run_ffn_half attn_half run_attn_half layer_d2048 run_layer_d2048 chain_dynscale run_chain_dynscale layer_mh run_layer_mh

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

# ---------- 2-CT single-column stub (Phase 1.5) ----------
${BUILD}/aie_gemm_pt_col_${M}x${K}x${N}.mlir: ${srcdir}/aie2_gemm_int8_srs_col.py
	mkdir -p ${@D}
	python3 $< -M ${M} -K ${K} -N ${N} > $@

${BUILD}/final_gemm_pt_col_${M}x${K}x${N}.xclbin: ${BUILD}/aie_gemm_pt_col_${M}x${K}x${N}.mlir \
                                                  ${BUILD}/llama_gemm_int8_srs_pt.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_gemm_pt_col_${M}x${K}x${N}.bin \
		$(notdir $<)

gemm_pt_col: ${BUILD}/final_gemm_pt_col_${M}x${K}x${N}.xclbin

run_gemm_pt_col: gemm_pt_col
	cd ${srcdir} && python3 test_gemm_int8_srs_pt_col.py \
		-x ${BUILD}/final_gemm_pt_col_${M}x${K}x${N}.xclbin \
		-i ${BUILD}/insts_gemm_pt_col_${M}x${K}x${N}.bin \
		-k MLIR_AIE -M ${M} -K ${K} -N ${N}

# ---------- 16-CT projection fan-out stub (Phase 1.6) ----------
${BUILD}/aie_gemm_pt_proj_${M}x${K}x${N}.mlir: ${srcdir}/aie2_gemm_int8_srs_proj.py
	mkdir -p ${@D}
	python3 $< -M ${M} -K ${K} -N ${N} > $@

${BUILD}/final_gemm_pt_proj_${M}x${K}x${N}.xclbin: ${BUILD}/aie_gemm_pt_proj_${M}x${K}x${N}.mlir \
                                                   ${BUILD}/llama_gemm_int8_srs_pt.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_gemm_pt_proj_${M}x${K}x${N}.bin \
		$(notdir $<)

gemm_pt_proj: ${BUILD}/final_gemm_pt_proj_${M}x${K}x${N}.xclbin

run_gemm_pt_proj: gemm_pt_proj
	cd ${srcdir} && python3 test_gemm_int8_srs_pt_proj.py \
		-x ${BUILD}/final_gemm_pt_proj_${M}x${K}x${N}.xclbin \
		-i ${BUILD}/insts_gemm_pt_proj_${M}x${K}x${N}.bin \
		-k MLIR_AIE -M ${M} -K ${K} -N ${N}

# ---------- FlowKV qk->sv pair stub (Phase 1.7) ----------
FLOWKV_BYTES ?= 512
FLOWKV_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_FLOWKV_PT_BYTES=${FLOWKV_BYTES}

${BUILD}/llama_flowkv_pt.cc.o: ${KDIR}/llama_flowkv_pt.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${FLOWKV_FLAGS} -c $< -o ${@F}

${BUILD}/aie_flowkv_pair_${FLOWKV_BYTES}.mlir: ${srcdir}/aie2_flowkv_pair.py
	mkdir -p ${@D}
	python3 $< --bytes ${FLOWKV_BYTES} > $@

${BUILD}/final_flowkv_pair_${FLOWKV_BYTES}.xclbin: ${BUILD}/aie_flowkv_pair_${FLOWKV_BYTES}.mlir \
                                                   ${BUILD}/llama_flowkv_pt.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_flowkv_pair_${FLOWKV_BYTES}.bin \
		$(notdir $<)

flowkv_pair: ${BUILD}/final_flowkv_pair_${FLOWKV_BYTES}.xclbin

run_flowkv_pair: flowkv_pair
	cd ${srcdir} && python3 test_flowkv_pair_pt.py \
		-x ${BUILD}/final_flowkv_pair_${FLOWKV_BYTES}.xclbin \
		-i ${BUILD}/insts_flowkv_pair_${FLOWKV_BYTES}.bin \
		-k MLIR_AIE --bytes ${FLOWKV_BYTES}

# ---------- 2-input glue stub (Phase 1.8) ----------
GLUE_BYTES ?= 512
GLUE_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_GLUE_PT_BYTES=${GLUE_BYTES}

${BUILD}/llama_glue_pt.cc.o: ${KDIR}/llama_glue_pt.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${GLUE_FLAGS} -c $< -o ${@F}

${BUILD}/aie_glue_${GLUE_BYTES}.mlir: ${srcdir}/aie2_glue.py
	mkdir -p ${@D}
	python3 $< --bytes ${GLUE_BYTES} > $@

${BUILD}/final_glue_${GLUE_BYTES}.xclbin: ${BUILD}/aie_glue_${GLUE_BYTES}.mlir \
                                          ${BUILD}/llama_glue_pt.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_glue_${GLUE_BYTES}.bin \
		$(notdir $<)

glue: ${BUILD}/final_glue_${GLUE_BYTES}.xclbin

run_glue: glue
	cd ${srcdir} && python3 test_glue_pt.py \
		-x ${BUILD}/final_glue_${GLUE_BYTES}.xclbin \
		-i ${BUILD}/insts_glue_${GLUE_BYTES}.bin \
		-k MLIR_AIE --bytes ${GLUE_BYTES}

# ---------- full single-layer integration stub (Phase 1.9) ----------
${BUILD}/llama_layer_pt.cc.o: ${KDIR}/llama_layer_pt.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}

${BUILD}/aie_layer_pt.mlir: ${srcdir}/aie2_layer.py
	mkdir -p ${@D}
	python3 $< > $@

${BUILD}/final_layer_pt.xclbin: ${BUILD}/aie_layer_pt.mlir ${BUILD}/llama_layer_pt.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_layer_pt.bin \
		$(notdir $<)

layer_pt: ${BUILD}/final_layer_pt.xclbin

run_layer_pt: layer_pt
	cd ${srcdir} && python3 test_layer_pt.py \
		-x ${BUILD}/final_layer_pt.xclbin \
		-i ${BUILD}/insts_layer_pt.bin \
		-k MLIR_AIE

# ---------- rmsnorm bf16 real kernel (Phase 2) ----------
RMS_D ?= 2048
RMS_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_RMSNORM_COLS=${RMS_D}

${BUILD}/llama_rmsnorm.cc.o: ${KDIR}/llama_rmsnorm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${RMS_FLAGS} -c $< -o ${@F}

${BUILD}/aie_rmsnorm_${RMS_D}.mlir: ${srcdir}/aie2_rmsnorm.py
	mkdir -p ${@D}
	python3 $< -D ${RMS_D} > $@

${BUILD}/final_rmsnorm_${RMS_D}.xclbin: ${BUILD}/aie_rmsnorm_${RMS_D}.mlir \
                                        ${BUILD}/llama_rmsnorm.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rmsnorm_${RMS_D}.bin \
		$(notdir $<)

rmsnorm: ${BUILD}/final_rmsnorm_${RMS_D}.xclbin

run_rmsnorm: rmsnorm
	cd ${srcdir} && python3 test_rmsnorm.py \
		-x ${BUILD}/final_rmsnorm_${RMS_D}.xclbin \
		-i ${BUILD}/insts_rmsnorm_${RMS_D}.bin \
		-k MLIR_AIE -D ${RMS_D}

# ---------- rmsnorm int8 real kernel (Phase 2) ----------
RMSI8_D ?= 2048
RMSI8_FLAGS := ${PEANOWRAP2P_FLAGS} -ffp-contract=off -DLLAMA_RMSNORM_COLS=${RMSI8_D}

# Per-shape rmsnorm .o files. The chain/layer designs use the D=64
# version (their fifos are 64-byte); the standalone rmsnorm test
# target uses whichever RMSI8_D the caller passes.
${BUILD}/llama_rmsnorm_int8_64.cc.o: ${KDIR}/llama_rmsnorm_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DLLAMA_RMSNORM_COLS=64 -c $< -o ${@F}

${BUILD}/llama_rmsnorm_int8.cc.o: ${KDIR}/llama_rmsnorm_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${RMSI8_FLAGS} -c $< -o ${@F}

${BUILD}/aie_rmsnorm_int8_${RMSI8_D}.mlir: ${srcdir}/aie2_rmsnorm_int8.py
	mkdir -p ${@D}
	python3 $< -D ${RMSI8_D} > $@

${BUILD}/final_rmsnorm_int8_${RMSI8_D}.xclbin: ${BUILD}/aie_rmsnorm_int8_${RMSI8_D}.mlir \
                                               ${BUILD}/llama_rmsnorm_int8.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rmsnorm_int8_${RMSI8_D}.bin \
		$(notdir $<)

rmsnorm_int8: ${BUILD}/final_rmsnorm_int8_${RMSI8_D}.xclbin

run_rmsnorm_int8: rmsnorm_int8
	cd ${srcdir} && python3 test_rmsnorm_int8.py \
		-x ${BUILD}/final_rmsnorm_int8_${RMSI8_D}.xclbin \
		-i ${BUILD}/insts_rmsnorm_int8_${RMSI8_D}.bin \
		-k MLIR_AIE -D ${RMSI8_D}

# Dynamic-output-scale variant (same .o; different IRON design).
${BUILD}/aie_rmsnorm_int8_dyn_${RMSI8_D}.mlir: ${srcdir}/aie2_rmsnorm_int8_dyn.py
	mkdir -p ${@D}
	python3 $< -D ${RMSI8_D} > $@

${BUILD}/final_rmsnorm_int8_dyn_${RMSI8_D}.xclbin: ${BUILD}/aie_rmsnorm_int8_dyn_${RMSI8_D}.mlir \
                                                   ${BUILD}/llama_rmsnorm_int8.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rmsnorm_int8_dyn_${RMSI8_D}.bin \
		$(notdir $<)

rmsnorm_int8_dyn: ${BUILD}/final_rmsnorm_int8_dyn_${RMSI8_D}.xclbin

run_rmsnorm_int8_dyn: rmsnorm_int8_dyn
	cd ${srcdir} && python3 test_rmsnorm_int8_dyn.py \
		-x ${BUILD}/final_rmsnorm_int8_dyn_${RMSI8_D}.xclbin \
		-i ${BUILD}/insts_rmsnorm_int8_dyn_${RMSI8_D}.bin \
		-k MLIR_AIE -D ${RMSI8_D}

# ---------- real gemm_int8_srs (Phase 2 GEMV path) ----------
# Shapes are hardcoded in the .cc (kD=64, kHD=256 for the layer
# integration test); 3 extern "C" symbols are emitted, plus a back-
# compat alias for the standalone bring-up.
GEMM_K ?= 64
GEMM_N ?= 64

${BUILD}/llama_gemm_int8_srs.cc.o: ${KDIR}/llama_gemm_int8_srs.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}

${BUILD}/aie_gemm_srs_${GEMM_K}x${GEMM_N}.mlir: ${srcdir}/aie2_gemm_int8_srs_real.py
	mkdir -p ${@D}
	python3 $< -K ${GEMM_K} -N ${GEMM_N} > $@

${BUILD}/final_gemm_srs_${GEMM_K}x${GEMM_N}.xclbin: ${BUILD}/aie_gemm_srs_${GEMM_K}x${GEMM_N}.mlir \
                                                    ${BUILD}/llama_gemm_int8_srs.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_gemm_srs_${GEMM_K}x${GEMM_N}.bin \
		$(notdir $<)

gemm_srs: ${BUILD}/final_gemm_srs_${GEMM_K}x${GEMM_N}.xclbin

run_gemm_srs: gemm_srs
	cd ${srcdir} && python3 test_gemm_int8_srs_real.py \
		-x ${BUILD}/final_gemm_srs_${GEMM_K}x${GEMM_N}.xclbin \
		-i ${BUILD}/insts_gemm_srs_${GEMM_K}x${GEMM_N}.bin \
		-k MLIR_AIE -K ${GEMM_K} -N ${GEMM_N}

# ---------- real silu_mul int8 (Phase 2) ----------
SILU_D ?= 8192
SILU_GATE_SCALE ?= 0.05
SILU_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_SILU_MUL_COLS=${SILU_D}

${BUILD}/silu_lut.h: ${srcdir}/gen_silu_lut.py
	mkdir -p ${@D}
	python3 $< --gate-scale ${SILU_GATE_SCALE} -o $@

${BUILD}/llama_silu_mul_int8.cc.o: ${KDIR}/llama_silu_mul_int8.cc ${BUILD}/silu_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${SILU_FLAGS} -I${srcdir} -c $< -o ${@F}

${BUILD}/aie_silu_mul_${SILU_D}.mlir: ${srcdir}/aie2_silu_mul_int8.py
	mkdir -p ${@D}
	python3 $< -D ${SILU_D} > $@

${BUILD}/final_silu_mul_${SILU_D}.xclbin: ${BUILD}/aie_silu_mul_${SILU_D}.mlir \
                                          ${BUILD}/llama_silu_mul_int8.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_silu_mul_${SILU_D}.bin \
		$(notdir $<)

silu_mul: ${BUILD}/final_silu_mul_${SILU_D}.xclbin

run_silu_mul: silu_mul
	cd ${srcdir} && python3 test_silu_mul_int8.py \
		-x ${BUILD}/final_silu_mul_${SILU_D}.xclbin \
		-i ${BUILD}/insts_silu_mul_${SILU_D}.bin \
		-k MLIR_AIE -D ${SILU_D}

# ---------- real rope int8 (Phase 2) ----------
ROPE_HD ?= 64
ROPE_NH ?= 8
ROPE_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_ROPE_HEAD_DIM=${ROPE_HD} -DLLAMA_ROPE_N_HEADS=${ROPE_NH}

${BUILD}/llama_rope_int8.cc.o: ${KDIR}/llama_rope_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${ROPE_FLAGS} -c $< -o ${@F}

${BUILD}/aie_rope_${ROPE_HD}x${ROPE_NH}.mlir: ${srcdir}/aie2_rope_int8.py
	mkdir -p ${@D}
	python3 $< --head-dim ${ROPE_HD} --n-heads ${ROPE_NH} > $@

${BUILD}/final_rope_${ROPE_HD}x${ROPE_NH}.xclbin: ${BUILD}/aie_rope_${ROPE_HD}x${ROPE_NH}.mlir \
                                                  ${BUILD}/llama_rope_int8.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rope_${ROPE_HD}x${ROPE_NH}.bin \
		$(notdir $<)

rope: ${BUILD}/final_rope_${ROPE_HD}x${ROPE_NH}.xclbin

run_rope: rope
	cd ${srcdir} && python3 test_rope_int8.py \
		-x ${BUILD}/final_rope_${ROPE_HD}x${ROPE_NH}.xclbin \
		-i ${BUILD}/insts_rope_${ROPE_HD}x${ROPE_NH}.bin \
		-k MLIR_AIE --head-dim ${ROPE_HD} --n-heads ${ROPE_NH}

# ---------- real flowkv attention pair (Phase 2) ----------
FKV_HD ?= 64
FKV_T  ?= 16
FKV_EXP_QUANT_SCALE ?= 0.05
FKV_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_FLOWKV_HEAD_DIM=${FKV_HD} -DLLAMA_FLOWKV_T=${FKV_T} -DLLAMA_FLOWKV_EXP_QUANT_SCALE=${FKV_EXP_QUANT_SCALE}f

${BUILD}/exp_lut.h: ${srcdir}/gen_exp_lut.py
	mkdir -p ${@D}
	python3 $< --exp-quant-scale ${FKV_EXP_QUANT_SCALE} -o $@

${BUILD}/llama_flowkv.cc.o: ${KDIR}/llama_flowkv.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${FKV_FLAGS} -I${srcdir} -c $< -o ${@F}

${BUILD}/aie_flowkv_${FKV_HD}x${FKV_T}.mlir: ${srcdir}/aie2_flowkv.py
	mkdir -p ${@D}
	python3 $< --head-dim ${FKV_HD} -T ${FKV_T} > $@

${BUILD}/final_flowkv_${FKV_HD}x${FKV_T}.xclbin: ${BUILD}/aie_flowkv_${FKV_HD}x${FKV_T}.mlir \
                                                 ${BUILD}/llama_flowkv.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_flowkv_${FKV_HD}x${FKV_T}.bin \
		$(notdir $<)

flowkv: ${BUILD}/final_flowkv_${FKV_HD}x${FKV_T}.xclbin

run_flowkv: flowkv
	cd ${srcdir} && python3 test_flowkv.py \
		-x ${BUILD}/final_flowkv_${FKV_HD}x${FKV_T}.xclbin \
		-i ${BUILD}/insts_flowkv_${FKV_HD}x${FKV_T}.bin \
		-k MLIR_AIE --head-dim ${FKV_HD} -T ${FKV_T}

# ---------- real sample (Phase 2 greedy v0) ----------
SAMPLE_V ?= 1024
SAMPLE_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_SAMPLE_VOCAB=${SAMPLE_V}

${BUILD}/llama_sample.cc.o: ${KDIR}/llama_sample.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${SAMPLE_FLAGS} -I${srcdir} -c $< -o ${@F}

${BUILD}/aie_sample_${SAMPLE_V}.mlir: ${srcdir}/aie2_sample.py
	mkdir -p ${@D}
	python3 $< -V ${SAMPLE_V} > $@

${BUILD}/final_sample_${SAMPLE_V}.xclbin: ${BUILD}/aie_sample_${SAMPLE_V}.mlir \
                                          ${BUILD}/llama_sample.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_sample_${SAMPLE_V}.bin \
		$(notdir $<)

sample: ${BUILD}/final_sample_${SAMPLE_V}.xclbin

run_sample: sample
	cd ${srcdir} && python3 test_sample.py \
		-x ${BUILD}/final_sample_${SAMPLE_V}.xclbin \
		-i ${BUILD}/insts_sample_${SAMPLE_V}.bin \
		-k MLIR_AIE -V ${SAMPLE_V}

# ---------- full single-layer integration with REAL kernels (Phase 3) ----------
# Sizes are hardcoded inside aie2_layer_real.py and the kernel .cc
# files (small first-bring-up: D=64, HD=256, head_dim=64, T=16).
# Layer integration reuses every Phase 2 kernel binary.

LAYER_REAL_DEPS = ${BUILD}/llama_rmsnorm_int8_64.cc.o \
                  ${BUILD}/llama_gemm_int8_srs.cc.o \
                  ${BUILD}/llama_rope_int8.cc.o \
                  ${BUILD}/llama_silu_mul_int8.cc.o \
                  ${BUILD}/llama_flowkv.cc.o \
                  ${BUILD}/llama_layer_pt.cc.o

# The layer/chain xclbins always link the D=64 rmsnorm. Override the
# linker artifact to be the D=64 build (the design feeds 64-byte
# buffers; the shape is hardcoded in the design via D=64).
${BUILD}/aie_layer_real.mlir ${BUILD}/aie_chain_real.mlir: RMSI8_D := 64

${BUILD}/aie_layer_real.mlir: ${srcdir}/aie2_layer_real.py
	mkdir -p ${@D}
	python3 $< > $@

${BUILD}/final_layer_real.xclbin: ${BUILD}/aie_layer_real.mlir ${LAYER_REAL_DEPS}
	mkdir -p ${@D}
	# llama_rmsnorm_int8.cc.o symlink to the D=64 build (chain/layer
	# design feeds 64-byte buffers; default D=2048 build silently
	# saturates the output).
	cp ${BUILD}/llama_rmsnorm_int8_64.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_layer_real.bin \
		$(notdir $<)

layer_real: ${BUILD}/final_layer_real.xclbin

run_layer_real: layer_real
	cd ${srcdir} && python3 test_layer_real.py \
		-x ${BUILD}/final_layer_real.xclbin \
		-i ${BUILD}/insts_layer_real.bin \
		-k MLIR_AIE

# ---------- N-layer decode chain (Phase 4) ----------
# Reuses every Phase 2 kernel binary; same set as layer_real plus the
# loop semantics + per-layer weight streaming.

CHAIN_DEPS = ${BUILD}/llama_rmsnorm_int8_64.cc.o \
             ${BUILD}/llama_gemm_int8_srs.cc.o \
             ${BUILD}/llama_rope_int8.cc.o \
             ${BUILD}/llama_silu_mul_int8.cc.o \
             ${BUILD}/llama_flowkv.cc.o \
             ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_chain_real.mlir: ${srcdir}/aie2_chain_real.py
	mkdir -p ${@D}
	python3 $< > $@

${BUILD}/final_chain_real.xclbin: ${BUILD}/aie_chain_real.mlir ${CHAIN_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_64.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_chain_real.bin \
		$(notdir $<)

chain_real: ${BUILD}/final_chain_real.xclbin

run_chain_real: chain_real
	cd ${srcdir} && python3 test_chain_real.py \
		-x ${BUILD}/final_chain_real.xclbin \
		-i ${BUILD}/insts_chain_real.bin \
		-k MLIR_AIE

# ---------- tiled decode GEMM (Phase 6c.1) ----------
# Weight-streamed GEMV: kernel processes N_TILE rows per call; IRON
# design loops N/N_TILE times. Activation kept resident in L1.
# N_TILE=4 keeps per-slot weight buffer (N_TILE*K = 8192 B) under one
# 16 KB L1 bank, avoiding Bug 3a's silent bank-aware fallback at
# N_TILE=8 (16416 B/slot spans two banks).
TILED_K      ?= 2048
TILED_N      ?= 64
TILED_N_TILE ?= 4

${BUILD}/llama_gemm_int8_srs_tiled_K${TILED_K}_t${TILED_N_TILE}.cc.o: ${KDIR}/llama_gemm_int8_srs_tiled.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DLLAMA_GEMM_TILED_K=${TILED_K} \
		-DLLAMA_GEMM_TILED_N_TILE=${TILED_N_TILE} -c $< -o ${@F}

${BUILD}/aie_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.mlir: ${srcdir}/aie2_gemm_tiled.py
	mkdir -p ${@D}
	python3 $< -K ${TILED_K} -N ${TILED_N} --n-tile ${TILED_N_TILE} > $@

${BUILD}/final_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.xclbin: \
		${BUILD}/aie_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.mlir \
		${BUILD}/llama_gemm_int8_srs_tiled_K${TILED_K}_t${TILED_N_TILE}.cc.o
	mkdir -p ${@D}
	cp $(word 2,$^) ${BUILD}/llama_gemm_int8_srs_tiled.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.bin \
		$(notdir $<)

tiled_gemm: ${BUILD}/final_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.xclbin

run_tiled_gemm: tiled_gemm
	cd ${srcdir} && python3 test_gemm_tiled.py \
		-x ${BUILD}/final_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.xclbin \
		-i ${BUILD}/insts_gemm_tiled_${TILED_K}x${TILED_N}_t${TILED_N_TILE}.bin \
		-k MLIR_AIE -K ${TILED_K} -N ${TILED_N} --n-tile ${TILED_N_TILE}

# ---------- FFN-half integration at D=2048, HD=8192 (Phase 6c.3b.1) ----------
# Six workers: rmsnorm2 + tiled gate + tiled up + silu_mul + tiled down + add2.
# Kernel object exposes BOTH K=2048 (gate/up) and K=8192 (down) extern
# entries in one .o so they link cleanly into one xclbin.
${BUILD}/llama_gemm_int8_srs_tiled_ffn.cc.o: ${KDIR}/llama_gemm_int8_srs_tiled_ffn.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}

# Per-shape kernel .o files. cp'd into the standard llama_*.cc.o name
# right before aiecc, same trick chain_real uses to avoid the chain's
# D=64 rmsnorm clobbering FFN-half's D=2048 build (and vice-versa).
${BUILD}/llama_rmsnorm_int8_2048.cc.o: ${KDIR}/llama_rmsnorm_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-ffp-contract=off \
		-DLLAMA_RMSNORM_COLS=2048 -c $< -o ${@F}

${BUILD}/llama_silu_mul_int8_8192.cc.o: ${KDIR}/llama_silu_mul_int8.cc ${BUILD}/silu_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DLLAMA_SILU_MUL_COLS=8192 -I${srcdir} -c $< -o ${@F}

FFN_HALF_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_ffn.cc.o \
                ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                ${BUILD}/llama_silu_mul_int8_8192.cc.o \
                ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_ffn_half.mlir: ${srcdir}/aie2_ffn_half.py
	mkdir -p ${@D}
	python3 $< > $@

# Diagnostic: rmsnorm + gate only, drain gf to host. Used to A/B against
# numpy gf when debugging the perchan FFN chain.
${BUILD}/aie_ffn_half_gate_only.mlir: ${srcdir}/aie2_ffn_half_gate_only.py ${srcdir}/aie2_ffn_half.py
	mkdir -p ${@D}
	python3 $< > $@

${BUILD}/final_ffn_half_gate_only.xclbin: ${BUILD}/aie_ffn_half_gate_only.mlir ${FFN_HALF_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_ffn_half_gate_only.bin \
		$(notdir $<)

ffn_half_gate_only: ${BUILD}/final_ffn_half_gate_only.xclbin

${BUILD}/final_ffn_half.xclbin: ${BUILD}/aie_ffn_half.mlir ${FFN_HALF_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_ffn_half.bin \
		$(notdir $<)

ffn_half: ${BUILD}/final_ffn_half.xclbin

run_ffn_half: ffn_half
	cd ${srcdir} && python3 test_ffn_half.py \
		-x ${BUILD}/final_ffn_half.xclbin \
		-i ${BUILD}/insts_ffn_half.bin \
		-k MLIR_AIE

# ---------- Per-channel-weight tiled GEMM standalone (Phase 6c.5b.1a) ----------
# Same kernel object as ffn_half (the per-channel extern entries live
# in llama_gemm_int8_srs_tiled_ffn.cc). Different design (aie2_gemm_
# tiled_perchan.py) that bakes per-call scales as Python closure
# constants. ACT/OUT scales come via env vars so the test can re-bake
# the xclbin for its own calibration.
TILED_PC_K      ?= 2048
TILED_PC_N      ?= 64
TILED_PC_N_TILE ?= 4
LLAMA_GEMM_ACT_SCALE     ?= 0.01
LLAMA_GEMM_INV_OUT_SCALE ?= 1.0

${BUILD}/aie_gemm_tiled_perchan_K${TILED_PC_K}_N${TILED_PC_N}_t${TILED_PC_N_TILE}.mlir: ${srcdir}/aie2_gemm_tiled_perchan.py
	mkdir -p ${@D}
	LLAMA_GEMM_ACT_SCALE=${LLAMA_GEMM_ACT_SCALE} \
	LLAMA_GEMM_INV_OUT_SCALE=${LLAMA_GEMM_INV_OUT_SCALE} \
	python3 $< -K ${TILED_PC_K} -N ${TILED_PC_N} --n-tile ${TILED_PC_N_TILE} > $@

${BUILD}/final_gemm_tiled_perchan_K${TILED_PC_K}_N${TILED_PC_N}_t${TILED_PC_N_TILE}.xclbin: \
		${BUILD}/aie_gemm_tiled_perchan_K${TILED_PC_K}_N${TILED_PC_N}_t${TILED_PC_N_TILE}.mlir \
		${BUILD}/llama_gemm_int8_srs_tiled_ffn.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_gemm_tiled_perchan_K${TILED_PC_K}_N${TILED_PC_N}_t${TILED_PC_N_TILE}.bin \
		$(notdir $<)

tiled_gemm_perchan: ${BUILD}/final_gemm_tiled_perchan_K${TILED_PC_K}_N${TILED_PC_N}_t${TILED_PC_N_TILE}.xclbin

run_tiled_gemm_perchan: tiled_gemm_perchan
	cd ${srcdir} && python3 test_gemm_tiled_perchan.py \
		-K ${TILED_PC_K} -N ${TILED_PC_N} --n-tile ${TILED_PC_N_TILE}

# ---------- Attention-half integration at D=2048, single-head (Phase 6c.3b.2) ----------
# Seven workers: rmsnorm1 + q_proj tiled + rope_q + flowkv pair + o_proj
# tiled + add1. Single-head simplification: N_HEADS=N_KV=1, HEAD_DIM=64.
# Host pre-fills the full KV cache.
ATTN_HALF_T ?= 128

${BUILD}/llama_gemm_int8_srs_tiled_attn.cc.o: ${KDIR}/llama_gemm_int8_srs_tiled_attn.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}

# Per-shape rope (N_HEADS=1, HEAD_DIM=64; matches defaults but stamped
# for cp-cleanness).
${BUILD}/llama_rope_int8_attn.cc.o: ${KDIR}/llama_rope_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-ffp-contract=off \
		-DLLAMA_ROPE_HEAD_DIM=64 -DLLAMA_ROPE_N_HEADS=1 -c $< -o ${@F}

# Per-T flowkv (HEAD_DIM=64, T set by ATTN_HALF_T).
${BUILD}/llama_flowkv_attn_T${ATTN_HALF_T}.cc.o: ${KDIR}/llama_flowkv.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DLLAMA_FLOWKV_HEAD_DIM=64 -DLLAMA_FLOWKV_T=${ATTN_HALF_T} \
		-DLLAMA_FLOWKV_EXP_QUANT_SCALE=0.05f -I${srcdir} -c $< -o ${@F}

ATTN_HALF_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_attn.cc.o \
                 ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                 ${BUILD}/llama_rope_int8_attn.cc.o \
                 ${BUILD}/llama_flowkv_attn_T${ATTN_HALF_T}.cc.o \
                 ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_attn_half_T${ATTN_HALF_T}.mlir: ${srcdir}/aie2_attn_half.py
	mkdir -p ${@D}
	LLAMA_ATTN_T=${ATTN_HALF_T} python3 $< > $@

${BUILD}/final_attn_half_T${ATTN_HALF_T}.xclbin: ${BUILD}/aie_attn_half_T${ATTN_HALF_T}.mlir ${ATTN_HALF_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_rope_int8_attn.cc.o    ${BUILD}/llama_rope_int8.cc.o
	cp ${BUILD}/llama_flowkv_attn_T${ATTN_HALF_T}.cc.o ${BUILD}/llama_flowkv.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_attn_half_T${ATTN_HALF_T}.bin \
		$(notdir $<)

attn_half: ${BUILD}/final_attn_half_T${ATTN_HALF_T}.xclbin

run_attn_half: attn_half
	cd ${srcdir} && LLAMA_ATTN_T=${ATTN_HALF_T} python3 test_attn_half.py \
		-x ${BUILD}/final_attn_half_T${ATTN_HALF_T}.xclbin \
		-i ${BUILD}/insts_attn_half_T${ATTN_HALF_T}.bin \
		-k MLIR_AIE

# ---------- Full single-layer at D=2048 (Phase 6c.3b.3) ----------
# 13 workers: FFN-half + attn-half combined into one xclbin. Reuses the
# per-shape rmsnorm/rope/flowkv .o files built by the FFN/attn rules.
LAYER_D2048_T ?= 128

${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o: ${KDIR}/llama_gemm_int8_srs_tiled_layer.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off -c $< -o ${@F}

LAYER_D2048_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o \
                   ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                   ${BUILD}/llama_silu_mul_int8_8192.cc.o \
                   ${BUILD}/llama_rope_int8_attn.cc.o \
                   ${BUILD}/llama_flowkv_attn_T${LAYER_D2048_T}.cc.o \
                   ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_layer_d2048_T${LAYER_D2048_T}.mlir: ${srcdir}/aie2_layer_d2048.py
	mkdir -p ${@D}
	LLAMA_LAYER_T=${LAYER_D2048_T} python3 $< > $@

${BUILD}/final_layer_d2048_T${LAYER_D2048_T}.xclbin: ${BUILD}/aie_layer_d2048_T${LAYER_D2048_T}.mlir ${LAYER_D2048_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_attn.cc.o ${BUILD}/llama_rope_int8.cc.o
	cp ${BUILD}/llama_flowkv_attn_T${LAYER_D2048_T}.cc.o ${BUILD}/llama_flowkv.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_layer_d2048_T${LAYER_D2048_T}.bin \
		$(notdir $<)

layer_d2048: ${BUILD}/final_layer_d2048_T${LAYER_D2048_T}.xclbin

run_layer_d2048: layer_d2048
	cd ${srcdir} && LLAMA_LAYER_T=${LAYER_D2048_T} python3 test_layer_d2048.py \
		-x ${BUILD}/final_layer_d2048_T${LAYER_D2048_T}.xclbin \
		-i ${BUILD}/insts_layer_d2048_T${LAYER_D2048_T}.bin \
		-k MLIR_AIE

# ---------- N-layer chain with perchan + dynamic scales (Phase 6c.5b.5) ----------
# Reuses the LAYER_D2048 kernel deps + the production-shape rmsnorm/silu/rope/flowkv .o
# files. N_LAYERS configurable via LLAMA_CHAIN_N env var (default 2 in the IRON design).
CHAIN_DYNSCALE_T ?= 128
CHAIN_DYNSCALE_N ?= 2

CHAIN_DYNSCALE_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o \
                      ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                      ${BUILD}/llama_silu_mul_int8_8192.cc.o \
                      ${BUILD}/llama_rope_int8_attn.cc.o \
                      ${BUILD}/llama_flowkv_attn_T${CHAIN_DYNSCALE_T}.cc.o \
                      ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.mlir: ${srcdir}/aie2_chain_dynscale.py
	mkdir -p ${@D}
	LLAMA_CHAIN_N=${CHAIN_DYNSCALE_N} LLAMA_CHAIN_T=${CHAIN_DYNSCALE_T} python3 $< > $@

${BUILD}/final_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.xclbin: ${BUILD}/aie_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.mlir ${CHAIN_DYNSCALE_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_attn.cc.o ${BUILD}/llama_rope_int8.cc.o
	cp ${BUILD}/llama_flowkv_attn_T${CHAIN_DYNSCALE_T}.cc.o ${BUILD}/llama_flowkv.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.bin \
		$(notdir $<)

chain_dynscale: ${BUILD}/final_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.xclbin

run_chain_dynscale: chain_dynscale
	cd ${srcdir} && LLAMA_CHAIN_N=${CHAIN_DYNSCALE_N} LLAMA_CHAIN_T=${CHAIN_DYNSCALE_T} python3 test_chain_dynscale.py \
		-x ${BUILD}/final_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.xclbin \
		-i ${BUILD}/insts_chain_dynscale_N${CHAIN_DYNSCALE_N}_T${CHAIN_DYNSCALE_T}.bin \
		-k MLIR_AIE

# ---------- chain + lm_head + sample (Phase 6b) ----------
# V is the lm_head/sample vocab. Both the gemm and sample .cc must be
# rebuilt with matching -DLLAMA_GEMM_VOCAB=$V / -DLLAMA_SAMPLE_VOCAB=$V.
# Use a V-stamped intermediate name and cp it into the standard
# llama_*.cc.o position aiecc looks for (mirrors the rmsnorm_int8_64 trick).
CHAIN_SAMPLE_V ?= 256

${BUILD}/llama_gemm_int8_srs_V${CHAIN_SAMPLE_V}.cc.o: ${KDIR}/llama_gemm_int8_srs.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DLLAMA_GEMM_VOCAB=${CHAIN_SAMPLE_V} -c $< -o ${@F}

${BUILD}/llama_sample_V${CHAIN_SAMPLE_V}.cc.o: ${KDIR}/llama_sample.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DLLAMA_SAMPLE_VOCAB=${CHAIN_SAMPLE_V} -I${srcdir} -c $< -o ${@F}

CHAIN_SAMPLE_DEPS = ${BUILD}/llama_rmsnorm_int8_64.cc.o \
                    ${BUILD}/llama_gemm_int8_srs_V${CHAIN_SAMPLE_V}.cc.o \
                    ${BUILD}/llama_rope_int8.cc.o \
                    ${BUILD}/llama_silu_mul_int8.cc.o \
                    ${BUILD}/llama_flowkv.cc.o \
                    ${BUILD}/llama_layer_pt.cc.o \
                    ${BUILD}/llama_sample_V${CHAIN_SAMPLE_V}.cc.o

${BUILD}/aie_chain_sample.mlir: ${srcdir}/aie2_chain_sample.py
	mkdir -p ${@D}
	LLAMA_CHAIN_SAMPLE_V=${CHAIN_SAMPLE_V} python3 $< > $@

${BUILD}/final_chain_sample.xclbin: ${BUILD}/aie_chain_sample.mlir ${CHAIN_SAMPLE_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_64.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_gemm_int8_srs_V${CHAIN_SAMPLE_V}.cc.o ${BUILD}/llama_gemm_int8_srs.cc.o
	cp ${BUILD}/llama_sample_V${CHAIN_SAMPLE_V}.cc.o ${BUILD}/llama_sample.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_chain_sample.bin \
		$(notdir $<)

chain_sample: ${BUILD}/final_chain_sample.xclbin

run_chain_sample: chain_sample
	cd ${srcdir} && LLAMA_CHAIN_SAMPLE_V=${CHAIN_SAMPLE_V} python3 test_chain_sample.py \
		-x ${BUILD}/final_chain_sample.xclbin \
		-i ${BUILD}/insts_chain_sample.bin \
		-k MLIR_AIE

# ---------- Single-layer multi-head GQA (Phase 7a) ----------
# Production multi-head shapes: 32 Q heads, 8 KV heads, REP=4, HEAD_DIM=64.
# 21 workers: 8 attn workers (one per KV head) + q_split + af_concat + the
# usual rmsnorm/q/o/gate/up/silu/down/add fabric.
LAYER_MH_T ?= 128

# Per-shape kernel objects.
${BUILD}/llama_rope_int8_mh_T${LAYER_MH_T}.cc.o: ${KDIR}/llama_rope_int8_mh.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-ffp-contract=off \
		-DLLAMA_ROPE_MH_HEAD_DIM=64 -DLLAMA_ROPE_MH_N_HEADS=32 -c $< -o ${@F}

${BUILD}/llama_flowkv_mh_T${LAYER_MH_T}.cc.o: ${KDIR}/llama_flowkv_mh.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_FLOWKV_HEAD_DIM=64 -DLLAMA_FLOWKV_T=${LAYER_MH_T} \
		-DLLAMA_FLOWKV_REP=4 -DLLAMA_FLOWKV_EXP_QUANT_SCALE=0.05f \
		-I${srcdir} -c $< -o ${@F}

${BUILD}/llama_gqa_glue.cc.o: ${KDIR}/llama_gqa_glue.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-DLLAMA_GQA_HEAD_DIM=64 -DLLAMA_GQA_N_HEADS_Q=32 \
		-DLLAMA_GQA_N_HEADS_KV=8 -c $< -o ${@F}

${BUILD}/llama_gemm_int8_srs_tiled_layer_mh.cc.o: ${KDIR}/llama_gemm_int8_srs_tiled_layer_mh.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off -c $< -o ${@F}

${BUILD}/llama_rescale_add.cc.o: ${KDIR}/llama_rescale_add.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off -DLLAMA_RESCALE_D=2048 -c $< -o ${@F}

LAYER_MH_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o \
                ${BUILD}/llama_gemm_int8_srs_tiled_layer_mh.cc.o \
                ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                ${BUILD}/llama_rescale_add.cc.o \
                ${BUILD}/llama_silu_mul_int8_8192.cc.o \
                ${BUILD}/llama_rope_int8_mh_T${LAYER_MH_T}.cc.o \
                ${BUILD}/llama_flowkv_mh_T${LAYER_MH_T}.cc.o \
                ${BUILD}/llama_gqa_glue.cc.o \
                ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_layer_mh_T${LAYER_MH_T}.mlir: ${srcdir}/aie2_layer_mh.py
	mkdir -p ${@D}
	LLAMA_LAYER_T=${LAYER_MH_T} python3 $< > $@

${BUILD}/final_layer_mh_T${LAYER_MH_T}.xclbin: ${BUILD}/aie_layer_mh_T${LAYER_MH_T}.mlir ${LAYER_MH_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${LAYER_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${LAYER_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_layer_mh_T${LAYER_MH_T}.bin \
		$(notdir $<)

layer_mh: ${BUILD}/final_layer_mh_T${LAYER_MH_T}.xclbin

run_layer_mh: layer_mh
	cd ${srcdir} && LLAMA_LAYER_T=${LAYER_MH_T} python3 test_layer_mh.py \
		-x ${BUILD}/final_layer_mh_T${LAYER_MH_T}.xclbin \
		-i ${BUILD}/insts_layer_mh_T${LAYER_MH_T}.bin \
		-k MLIR_AIE

# ---------- N-layer multi-head GQA chain (Phase 7b) ----------
CHAIN_MH_T ?= 128
CHAIN_MH_N ?= 2

CHAIN_MH_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o \
                ${BUILD}/llama_gemm_int8_srs_tiled_layer_mh.cc.o \
                ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                ${BUILD}/llama_silu_mul_int8_8192.cc.o \
                ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o \
                ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o \
                ${BUILD}/llama_gqa_glue.cc.o \
                ${BUILD}/llama_layer_pt.cc.o

${BUILD}/aie_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.mlir: ${srcdir}/aie2_chain_dynscale_mh.py
	mkdir -p ${@D}
	LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< > $@

${BUILD}/final_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin: ${BUILD}/aie_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.mlir ${CHAIN_MH_DEPS}
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin \
		$(notdir $<)

chain_mh: ${BUILD}/final_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin

run_chain_mh: chain_mh
	cd ${srcdir} && LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_mh.py \
		-x ${BUILD}/final_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin \
		-i ${BUILD}/insts_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin \
		-k MLIR_AIE

# ---------- clean ----------
clean:
	rm -rf ${BUILD}
