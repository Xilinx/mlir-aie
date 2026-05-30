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

.PHONY: all clean gemm_pt run_gemm_pt gemm_pt_col run_gemm_pt_col gemm_pt_proj run_gemm_pt_proj flowkv_pair run_flowkv_pair glue run_glue layer_pt run_layer_pt rmsnorm run_rmsnorm rmsnorm_int8 run_rmsnorm_int8 gemm_srs run_gemm_srs silu_mul run_silu_mul rope run_rope flowkv run_flowkv sample run_sample layer_real run_layer_real chain_real run_chain_real

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
RMSI8_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_RMSNORM_COLS=${RMSI8_D}

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

${BUILD}/llama_sample.cc.o: ${KDIR}/llama_sample.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${SAMPLE_FLAGS} -c $< -o ${@F}

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

# ---------- clean ----------
clean:
	rm -rf ${BUILD}
