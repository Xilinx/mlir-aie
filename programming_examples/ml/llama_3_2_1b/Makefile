##===- Makefile -----------------------------------------------------------===##
#
# Llama 3.2 1B decode on NPU2: the single multi-head layer (layer_mh), the
# N-layer decode chain in each of its modes (chain_mh, chain_sample_mh,
# chain_gather_mh, chain_onestream_mh, chain_persist_mh, chain_persist_grow_mh,
# chain_persist_resident_mh), the per-kernel unit designs (rmsnorm_int8,
# rmsnorm_int8_dyn, silu_mul, rope, flowkv) and the top-k sampler probe. Each
# design has a build target and a run_<target> that builds it and checks it on
# the NPU against its numpy reference; README.md has the usual invocations.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

BUILD := ${srcdir}/build
KDIR  := ${srcdir}/kernels

.PHONY: all clean \
	rmsnorm_int8 run_rmsnorm_int8 rmsnorm_int8_dyn run_rmsnorm_int8_dyn \
	silu_mul run_silu_mul rope run_rope flowkv run_flowkv \
	layer_mh run_layer_mh chain_mh run_chain_mh chain_sample_mh run_chain_sample_mh \
	chain_gather_mh chain_onestream_mh run_chain_onestream_mh \
	chain_persist_mh run_chain_persist_mh chain_persist_grow_mh run_chain_persist_grow_mh \
	chain_persist_resident_mh run_chain_persist_resident_mh \
	topk_sample_probe run_topk_sample_probe

# The designs the PR-time lit tests build.
all: layer_mh chain_mh chain_persist_grow_mh

# ---------- rmsnorm int8 real kernel (Phase 2) ----------
RMSI8_D ?= 2048

# Per-shape rmsnorm .o files, named by the COLS they were compiled with so
# Make's dependency tracking (and the chain/layer designs' cp-before-use
# trick) can't silently reuse an object built for a different D. The D=64
# and D=2048 shapes have their own explicit rules (Make prefers an explicit
# rule over a pattern rule); this pattern rule covers any other D the
# standalone rmsnorm test is asked to build.
${BUILD}/llama_rmsnorm_int8_64.cc.o: ${KDIR}/llama_rmsnorm_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DLLAMA_RMSNORM_COLS=64 -c $< -o ${@F}

${BUILD}/llama_rmsnorm_int8_%.cc.o: ${KDIR}/llama_rmsnorm_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_RMSNORM_COLS=$* -c $< -o ${@F}

${BUILD}/aie_rmsnorm_int8_${RMSI8_D}.mlir: ${srcdir}/aie2_rmsnorm_int8.py
	mkdir -p ${@D}
	python3 $< -D ${RMSI8_D} > $@

${BUILD}/final_rmsnorm_int8_${RMSI8_D}.xclbin: ${BUILD}/aie_rmsnorm_int8_${RMSI8_D}.mlir \
                                               ${BUILD}/llama_rmsnorm_int8_${RMSI8_D}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_${RMSI8_D}.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rmsnorm_int8_${RMSI8_D}.bin \
		$(notdir $<)

rmsnorm_int8: ${BUILD}/final_rmsnorm_int8_${RMSI8_D}.xclbin

run_rmsnorm_int8: rmsnorm_int8
	cd ${srcdir} && python3 test_rmsnorm_int8.py \
		--xclbin ${BUILD}/final_rmsnorm_int8_${RMSI8_D}.xclbin \
		--instr ${BUILD}/insts_rmsnorm_int8_${RMSI8_D}.bin \
		-k MLIR_AIE -D ${RMSI8_D}

# Dynamic-output-scale variant (same .o; different IRON design).
${BUILD}/aie_rmsnorm_int8_dyn_${RMSI8_D}.mlir: ${srcdir}/aie2_rmsnorm_int8_dyn.py
	mkdir -p ${@D}
	python3 $< -D ${RMSI8_D} > $@

${BUILD}/final_rmsnorm_int8_dyn_${RMSI8_D}.xclbin: ${BUILD}/aie_rmsnorm_int8_dyn_${RMSI8_D}.mlir \
                                                   ${BUILD}/llama_rmsnorm_int8_${RMSI8_D}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_rmsnorm_int8_${RMSI8_D}.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rmsnorm_int8_dyn_${RMSI8_D}.bin \
		$(notdir $<)

rmsnorm_int8_dyn: ${BUILD}/final_rmsnorm_int8_dyn_${RMSI8_D}.xclbin

run_rmsnorm_int8_dyn: rmsnorm_int8_dyn
	cd ${srcdir} && python3 test_rmsnorm_int8_dyn.py \
		--xclbin ${BUILD}/final_rmsnorm_int8_dyn_${RMSI8_D}.xclbin \
		--instr ${BUILD}/insts_rmsnorm_int8_dyn_${RMSI8_D}.bin \
		-k MLIR_AIE -D ${RMSI8_D}

# ---------- real silu_mul int8 (Phase 2) ----------
SILU_D ?= 8192
SILU_GATE_SCALE ?= 0.05
# SILU_SELFCAL picks aie2_silu_mul_int8.py's dyn vs. selfcal IRON design
# (the two bind different kernel entry points with different I/O shapes).
# It must be baked into the generated .mlir/.xclbin filename, not just
# exported to the subprocess env, or Make's mtime-based dependency
# tracking can't tell the two modes apart and silently reuses an xclbin
# built for the other one.
SILU_SELFCAL ?= 0
SILU_TAG := $(if $(filter 1,${SILU_SELFCAL}),_selfcal,)

${BUILD}/silu_lut.h: ${srcdir}/gen_silu_lut.py
	mkdir -p ${@D}
	python3 $< --gate-scale ${SILU_GATE_SCALE} -o $@

# D=8192 (HIDDEN_DIM) has its own explicit rule, matching the chain/layer
# designs' cp-before-use object (Make prefers an explicit rule over a
# pattern rule); this pattern rule covers any other D.
${BUILD}/llama_silu_mul_int8_%.cc.o: ${KDIR}/llama_silu_mul_int8.cc ${BUILD}/silu_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_SILU_MUL_COLS=$* -I${srcdir} -c $< -o ${@F}

${BUILD}/aie_silu_mul_${SILU_D}${SILU_TAG}.mlir: ${srcdir}/aie2_silu_mul_int8.py
	mkdir -p ${@D}
	SILU_SELFCAL=${SILU_SELFCAL} python3 $< -D ${SILU_D} > $@

${BUILD}/final_silu_mul_${SILU_D}${SILU_TAG}.xclbin: ${BUILD}/aie_silu_mul_${SILU_D}${SILU_TAG}.mlir \
                                          ${BUILD}/llama_silu_mul_int8_${SILU_D}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_silu_mul_int8_${SILU_D}.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_silu_mul_${SILU_D}${SILU_TAG}.bin \
		$(notdir $<)

silu_mul: ${BUILD}/final_silu_mul_${SILU_D}${SILU_TAG}.xclbin

run_silu_mul: silu_mul
	cd ${srcdir} && SILU_SELFCAL=${SILU_SELFCAL} python3 test_silu_mul_int8.py \
		--xclbin ${BUILD}/final_silu_mul_${SILU_D}${SILU_TAG}.xclbin \
		--instr ${BUILD}/insts_silu_mul_${SILU_D}${SILU_TAG}.bin \
		-k MLIR_AIE -D ${SILU_D}

# ---------- real rope int8 (Phase 2) ----------
ROPE_HD ?= 64
ROPE_NH ?= 8
ROPE_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_ROPE_HEAD_DIM=${ROPE_HD} -DLLAMA_ROPE_N_HEADS=${ROPE_NH}
# ROPE_DYN picks llama_rope_int8 (static scale) vs. llama_rope_int8_dyn
# (reads/passes through act_scale via a tail); baked into the filename for
# the same reason SILU_SELFCAL is (see the silu_mul section above).
ROPE_DYN ?= 0
ROPE_TAG := $(if $(filter 1,${ROPE_DYN}),_dyn,)

${BUILD}/llama_rope_int8_${ROPE_HD}x${ROPE_NH}.cc.o: ${KDIR}/llama_rope_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${ROPE_FLAGS} -c $< -o ${@F}

${BUILD}/aie_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.mlir: ${srcdir}/aie2_rope_int8.py
	mkdir -p ${@D}
	ROPE_DYN=${ROPE_DYN} python3 $< --head-dim ${ROPE_HD} --n-heads ${ROPE_NH} > $@

${BUILD}/final_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.xclbin: ${BUILD}/aie_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.mlir \
                                                  ${BUILD}/llama_rope_int8_${ROPE_HD}x${ROPE_NH}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_rope_int8_${ROPE_HD}x${ROPE_NH}.cc.o ${BUILD}/llama_rope_int8.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.bin \
		$(notdir $<)

rope: ${BUILD}/final_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.xclbin

run_rope: rope
	cd ${srcdir} && ROPE_DYN=${ROPE_DYN} python3 test_rope_int8.py \
		--xclbin ${BUILD}/final_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.xclbin \
		--instr ${BUILD}/insts_rope_${ROPE_HD}x${ROPE_NH}${ROPE_TAG}.bin \
		-k MLIR_AIE --head-dim ${ROPE_HD} --n-heads ${ROPE_NH}

# ---------- real flowkv attention pair (Phase 2) ----------
FKV_HD ?= 64
FKV_T  ?= 16
FKV_EXP_QUANT_SCALE ?= 0.05
FKV_FLAGS := ${PEANOWRAP2P_FLAGS} -DLLAMA_FLOWKV_HEAD_DIM=${FKV_HD} -DLLAMA_FLOWKV_T=${FKV_T} -DLLAMA_FLOWKV_EXP_QUANT_SCALE=${FKV_EXP_QUANT_SCALE}f
# FLOWKV_DYN picks llama_flowkv_{qk,sv} (static scale) vs.
# llama_flowkv_{qk,sv}_dyn (scales read from tails); baked into the
# filename for the same reason SILU_SELFCAL/ROPE_DYN are.
FLOWKV_DYN ?= 0
FLOWKV_TAG := $(if $(filter 1,${FLOWKV_DYN}),_dyn,)

${BUILD}/exp_lut.h: ${srcdir}/gen_exp_lut.py
	mkdir -p ${@D}
	python3 $< --exp-quant-scale ${FKV_EXP_QUANT_SCALE} -o $@

${BUILD}/llama_flowkv_${FKV_HD}x${FKV_T}.cc.o: ${KDIR}/llama_flowkv.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${FKV_FLAGS} -I${srcdir} -c $< -o ${@F}

${BUILD}/aie_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.mlir: ${srcdir}/aie2_flowkv.py
	mkdir -p ${@D}
	FLOWKV_DYN=${FLOWKV_DYN} python3 $< --head-dim ${FKV_HD} -T ${FKV_T} > $@

${BUILD}/final_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.xclbin: ${BUILD}/aie_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.mlir \
                                                 ${BUILD}/llama_flowkv_${FKV_HD}x${FKV_T}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_flowkv_${FKV_HD}x${FKV_T}.cc.o ${BUILD}/llama_flowkv.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.bin \
		$(notdir $<)

flowkv: ${BUILD}/final_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.xclbin

run_flowkv: flowkv
	cd ${srcdir} && FLOWKV_DYN=${FLOWKV_DYN} python3 test_flowkv.py \
		--xclbin ${BUILD}/final_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.xclbin \
		--instr ${BUILD}/insts_flowkv_${FKV_HD}x${FKV_T}${FLOWKV_TAG}.bin \
		-k MLIR_AIE --head-dim ${FKV_HD} -T ${FKV_T}

# ---------- Production-shape kernel objects shared by the layer/chain targets ----------
# Per-shape .o files, cp'd into the standard llama_*.cc.o name right before
# aiecc so differently-shaped builds of the same kernel don't clobber each other.
${BUILD}/llama_rmsnorm_int8_2048.cc.o: ${KDIR}/llama_rmsnorm_int8.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-ffp-contract=off \
		-DLLAMA_RMSNORM_COLS=2048 -c $< -o ${@F}

${BUILD}/llama_silu_mul_int8_8192.cc.o: ${KDIR}/llama_silu_mul_int8.cc ${BUILD}/silu_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_SILU_MUL_COLS=8192 -I${srcdir} -c $< -o ${@F}

# LMHEAD_KSET sizes the fused lm_head+top-k insert's resident set capacity; it
# only affects llama_lmhead_topk_insert (unused by the layer/chain GEMMs, so the
# default 16 leaves them unchanged). Each KSET is its own object, and a target
# cp's the one it needs to the basename its Kernel() op references.
LMHEAD_KSET ?= 16
${BUILD}/llama_gemm_int8_srs_tiled_layer_k%.cc.o: ${KDIR}/llama_gemm_int8_srs_tiled_layer.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_LMHEAD_KSET=$* -c $< -o ${@F}

# Shared by the layer and chain targets.
KVA_T ?= 128
${BUILD}/llama_kv_append.cc.o: ${KDIR}/llama_kv_append.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_KVA_HEAD_DIM=64 -DLLAMA_KVA_T=${KVA_T} -c $< -o ${@F}

${BUILD}/llama_layer_pt.cc.o: ${KDIR}/llama_layer_pt.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}

# The streamed sampler's vocab and chunk, and the sampler params the
# chain-sample modes bake in.
SAMPLE_STREAM_V ?= 128256
SAMPLE_STREAM_CHUNK ?= 2004
${BUILD}/llama_sample_streamed.cc.o: ${KDIR}/llama_sample_streamed.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
		-ffp-contract=off -DLLAMA_SAMPLE_EXP_QUANT_SCALE=0.05f \
		-DLLAMA_SAMPLE_VOCAB=${SAMPLE_STREAM_V} \
		-DLLAMA_SAMPLE_CHUNK_N=${SAMPLE_STREAM_CHUNK} \
		-I${srcdir} -c $< -o ${@F}

LMFULL_TEMP ?= 0.0
LMFULL_TOPK ?= 0
LMFULL_SEED ?= 0

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

LAYER_MH_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o \
                ${BUILD}/llama_gemm_int8_srs_tiled_layer_mh.cc.o \
                ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                ${BUILD}/llama_rescale_add.cc.o \
                ${BUILD}/llama_kv_append.cc.o \
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
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${LAYER_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${LAYER_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_layer_mh_T${LAYER_MH_T}.bin \
		$(notdir $<)

layer_mh: ${BUILD}/final_layer_mh_T${LAYER_MH_T}.xclbin

run_layer_mh: layer_mh
	cd ${srcdir} && LLAMA_LAYER_T=${LAYER_MH_T} python3 test_layer_mh.py \
		--xclbin ${BUILD}/final_layer_mh_T${LAYER_MH_T}.xclbin \
		--instr ${BUILD}/insts_layer_mh_T${LAYER_MH_T}.bin \
		-k MLIR_AIE

# ---------- N-layer multi-head GQA chain (Phase 7b) ----------
CHAIN_MH_T ?= 128
CHAIN_MH_N ?= 2

CHAIN_MH_DEPS = ${BUILD}/llama_gemm_int8_srs_tiled_layer_mh.cc.o \
                ${BUILD}/llama_rmsnorm_int8_2048.cc.o \
                ${BUILD}/llama_rescale_add.cc.o \
                ${BUILD}/llama_kv_append.cc.o \
                ${BUILD}/llama_silu_mul_int8_8192.cc.o \
                ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o \
                ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o \
                ${BUILD}/llama_gqa_glue.cc.o \
                ${BUILD}/llama_layer_pt.cc.o

${BUILD}/final_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${CHAIN_MH_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${BUILD} && LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin

chain_mh: ${BUILD}/final_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin

run_chain_mh: chain_mh
	cd ${srcdir} && LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_mh.py \
		--xclbin ${BUILD}/final_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin \
		--instr ${BUILD}/insts_chain_mh_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin \
		-k MLIR_AIE

# ---------- M3b: chain + on-chip lm_head + sampler (LLAMA_CHAIN_SAMPLE=1) ----------
CHAIN_SAMPLE_DEPS = ${CHAIN_MH_DEPS} ${BUILD}/llama_sample_streamed.cc.o

${BUILD}/final_chain_sample_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${srcdir}/logits_relay.py ${CHAIN_SAMPLE_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${BUILD} && LLAMA_CHAIN_SAMPLE=1 LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_sample_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin

chain_sample_mh: ${BUILD}/final_chain_sample_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin

run_chain_sample_mh: chain_sample_mh
	cd ${srcdir} && LLAMA_CHAIN_SAMPLE=1 LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_sample_mh.py \
		--xclbin ${BUILD}/final_chain_sample_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin \
		--instr ${BUILD}/insts_chain_sample_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin \
		-k MLIR_AIE

# ---------- M3c: chain + lm_head + sampler + on-chip embed gather (token->token) ----------
# embed_select compiled at the real chain D=2048 (the probe uses ES_D=256).
${BUILD}/llama_embed_select_2048.cc.o: ${KDIR}/llama_embed_select.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-DLLAMA_EMBED_D=2048 -DLLAMA_EMBED_NTILE=4 -DLLAMA_EMBED_PREFIX=64 -c $< -o ${@F}

CHAIN_GATHER_DEPS = ${CHAIN_SAMPLE_DEPS} ${BUILD}/llama_embed_select_2048.cc.o

${BUILD}/final_chain_gather_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${srcdir}/logits_relay.py ${CHAIN_GATHER_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${LMHEAD_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cp ${BUILD}/llama_embed_select_2048.cc.o ${BUILD}/llama_embed_select.cc.o
	cd ${BUILD} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_GATHER=1 LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_gather_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin

chain_gather_mh: ${BUILD}/final_chain_gather_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin

# ---------- one-stream sampler+gather fused into the chain (LLAMA_CHAIN_ONESTREAM=1) ----------
# The fused lm_head GEMM + top-k insert (the gemm object at KSET) and
# llama_topk_finalize (the topk object at D=2048, KSET) must agree on KSET
# (== ONESTREAM_KSET).
ONESTREAM_KSET ?= 8
CHAIN_ONESTREAM_DEPS = ${CHAIN_MH_DEPS} ${BUILD}/llama_sample_streamed.cc.o

${BUILD}/final_chain_onestream_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${CHAIN_ONESTREAM_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${BUILD} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_onestream_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin

chain_onestream_mh: ${BUILD}/final_chain_onestream_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin

run_chain_onestream_mh: chain_onestream_mh
	cd ${srcdir} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_onestream_mh.py \
		--xclbin ${BUILD}/final_chain_onestream_N${CHAIN_MH_N}_T${CHAIN_MH_T}.xclbin \
		--instr ${BUILD}/insts_chain_onestream_N${CHAIN_MH_N}_T${CHAIN_MH_T}.bin \
		-k MLIR_AIE

# ---------- persistent on-device autoregressive loop (LLAMA_CHAIN_PERSIST=1) ----------
# PT tokens in ONE dispatch, the sampled token's embed seed feeding back ON-CHIP
# (no host between tokens). Increment 1: KV at fixed position (host re-streams the
# pristine KV each token). Reuses the onestream deps + kernel objects.
PT ?= 4
CHAIN_PERSIST_DEPS = ${CHAIN_MH_DEPS} ${BUILD}/llama_sample_streamed.cc.o

${BUILD}/final_chain_persist_N${CHAIN_MH_N}_PT${PT}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${CHAIN_PERSIST_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${BUILD} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_PERSIST=1 LLAMA_CHAIN_PT=${PT} \
		LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_persist_N${CHAIN_MH_N}_PT${PT}.bin

chain_persist_mh: ${BUILD}/final_chain_persist_N${CHAIN_MH_N}_PT${PT}.xclbin

run_chain_persist_mh: chain_persist_mh
	cd ${srcdir} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_PERSIST=1 LLAMA_CHAIN_PT=${PT} \
		LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_persist_mh.py \
		--xclbin ${BUILD}/final_chain_persist_N${CHAIN_MH_N}_PT${PT}.xclbin \
		--instr ${BUILD}/insts_chain_persist_N${CHAIN_MH_N}_PT${PT}.bin \
		-k MLIR_AIE

# ---------- persistent loop with GROWING KV (increment 2, LLAMA_CHAIN_PERSIST_GROW=1) ----------
# KV cache accumulates across the PT tokens (real autoregressive decode) via the
# on-chip advancing append + host KV ping-pong. Distinct artifact names so it
# doesn't collide with the fixed-position increment-1 build.
${BUILD}/final_chain_persist_grow_N${CHAIN_MH_N}_PT${PT}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${CHAIN_PERSIST_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${BUILD} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_PERSIST=1 LLAMA_CHAIN_PERSIST_GROW=1 \
		LLAMA_CHAIN_PT=${PT} LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_persist_grow_N${CHAIN_MH_N}_PT${PT}.bin

chain_persist_grow_mh: ${BUILD}/final_chain_persist_grow_N${CHAIN_MH_N}_PT${PT}.xclbin

# Extra test flags, e.g. PERSIST_GROW_ARGS=--synthetic-weights (no data/ needed).
PERSIST_GROW_ARGS ?=

run_chain_persist_grow_mh: chain_persist_grow_mh
	cd ${srcdir} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_PERSIST=1 LLAMA_CHAIN_PERSIST_GROW=1 \
		LLAMA_CHAIN_PT=${PT} LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_persist_grow_mh.py \
		--xclbin ${BUILD}/final_chain_persist_grow_N${CHAIN_MH_N}_PT${PT}.xclbin \
		--instr ${BUILD}/insts_chain_persist_grow_N${CHAIN_MH_N}_PT${PT}.bin \
		-k MLIR_AIE ${PERSIST_GROW_ARGS}

# ---------- persistent loop with RESIDENT KV (LLAMA_CHAIN_PERSIST_RESIDENT=1) ----------
# KV cache resident on the attn compute tiles (b_kv); host streams ONLY weights
# after the token-0 seed. Scoped to small N (N=2 -> 32KB/head fits the tile).
${BUILD}/final_chain_persist_resident_N${CHAIN_MH_N}_PT${PT}.xclbin: ${srcdir}/aie2_chain_dynscale_mh.py ${CHAIN_PERSIST_DEPS} ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_gemm_int8_srs_tiled_layer_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_gemm_int8_srs_tiled_layer.cc.o
	cp ${BUILD}/llama_topk_sample_d2048_k${ONESTREAM_KSET}.cc.o ${BUILD}/llama_topk_sample.cc.o
	cp ${BUILD}/llama_rmsnorm_int8_2048.cc.o ${BUILD}/llama_rmsnorm_int8.cc.o
	cp ${BUILD}/llama_silu_mul_int8_8192.cc.o ${BUILD}/llama_silu_mul_int8.cc.o
	cp ${BUILD}/llama_rope_int8_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_rope_int8_mh.cc.o
	cp ${BUILD}/llama_flowkv_mh_T${CHAIN_MH_T}.cc.o ${BUILD}/llama_flowkv_mh.cc.o
	cd ${BUILD} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_PERSIST=1 LLAMA_CHAIN_PERSIST_GROW=1 \
		LLAMA_CHAIN_PERSIST_RESIDENT=1 \
		LLAMA_CHAIN_PT=${PT} LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 $< \
		--xclbin-path $@ \
		--insts-path ${BUILD}/insts_chain_persist_resident_N${CHAIN_MH_N}_PT${PT}.bin

chain_persist_resident_mh: ${BUILD}/final_chain_persist_resident_N${CHAIN_MH_N}_PT${PT}.xclbin

run_chain_persist_resident_mh: chain_persist_resident_mh
	cd ${srcdir} && LLAMA_CHAIN_SAMPLE=1 LLAMA_CHAIN_ONESTREAM=1 LLAMA_CHAIN_PERSIST=1 LLAMA_CHAIN_PERSIST_GROW=1 \
		LLAMA_CHAIN_PERSIST_RESIDENT=1 \
		LLAMA_CHAIN_PT=${PT} LLAMA_CHAIN_ONESTREAM_KSET=${ONESTREAM_KSET} \
		LLAMA_SAMPLE_TEMP=${LMFULL_TEMP} LLAMA_SAMPLE_TOPK=${LMFULL_TOPK} LLAMA_SAMPLE_SEED=${LMFULL_SEED} \
		LLAMA_CHAIN_N=${CHAIN_MH_N} LLAMA_CHAIN_T=${CHAIN_MH_T} python3 test_chain_persist_resident_mh.py \
		--xclbin ${BUILD}/final_chain_persist_resident_N${CHAIN_MH_N}_PT${PT}.xclbin \
		--instr ${BUILD}/insts_chain_persist_resident_N${CHAIN_MH_N}_PT${PT}.bin \
		-k MLIR_AIE

# ---------- one-stream top-k sampler + embed gather probe ----------
TK_V ?= 64
TK_D ?= 256
TK_KSET ?= 8
TK_TEMP ?= 0.0
TK_TOPK ?= 8
TK_SEED ?= 0
# The sampler params (temp/top_k/seed) are BAKED into the buffer at codegen, so
# the MLIR + xclbin are config-specific. Encode the config in the artifact names
# so a different temp/top_k/seed never silently reuses a stale build.
TK_CFG := V${TK_V}_D${TK_D}_K${TK_KSET}_t${TK_TEMP}_k${TK_TOPK}_s${TK_SEED}
# Each (TK_D, TK_KSET) is its own object, llama_topk_sample_d<D>_k<KSET>.cc.o; a
# target cp's the one it needs to llama_topk_sample.cc.o, the basename its
# Kernel() op references.
${BUILD}/llama_topk_sample_d%.cc.o: ${KDIR}/llama_topk_sample.cc ${BUILD}/exp_lut.h
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -ffp-contract=off \
		-I${srcdir} -DLLAMA_TOPK_D=$(word 1,$(subst _k, ,$*)) -DLLAMA_TOPK_NTILE=4 \
		-DLLAMA_TOPK_KSET=$(word 2,$(subst _k, ,$*)) -c $< -o ${@F}

${BUILD}/aie_topk_sample_probe_${TK_CFG}.mlir: ${srcdir}/aie2_topk_sample_probe.py
	mkdir -p ${@D}
	LLAMA_TK_V=${TK_V} LLAMA_TK_D=${TK_D} LLAMA_TK_KSET=${TK_KSET} \
		LLAMA_TK_TEMP=${TK_TEMP} LLAMA_TK_TOPK=${TK_TOPK} LLAMA_TK_SEED=${TK_SEED} python3 $< > $@

${BUILD}/final_topk_sample_probe_${TK_CFG}.xclbin: ${BUILD}/aie_topk_sample_probe_${TK_CFG}.mlir ${BUILD}/llama_topk_sample_d${TK_D}_k${TK_KSET}.cc.o
	mkdir -p ${@D}
	cp ${BUILD}/llama_topk_sample_d${TK_D}_k${TK_KSET}.cc.o ${BUILD}/llama_topk_sample.cc.o
	cd ${@D} && aiecc --unified --peano=${PEANO_INSTALL_DIR} --get-xclbin --get-npu-insts \
		--xclbin-name=${@F} \
		--npu-insts-name=insts_topk_sample_probe_${TK_CFG}.bin \
		$(notdir $<)

topk_sample_probe: ${BUILD}/final_topk_sample_probe_${TK_CFG}.xclbin

run_topk_sample_probe: topk_sample_probe
	cd ${srcdir} && LLAMA_TK_V=${TK_V} LLAMA_TK_D=${TK_D} LLAMA_TK_KSET=${TK_KSET} \
		LLAMA_TK_TEMP=${TK_TEMP} LLAMA_TK_TOPK=${TK_TOPK} LLAMA_TK_SEED=${TK_SEED} \
		python3 test_topk_sample_probe.py \
		--xclbin ${BUILD}/final_topk_sample_probe_${TK_CFG}.xclbin \
		--instr ${BUILD}/insts_topk_sample_probe_${TK_CFG}.bin \
		-k MLIR_AIE

# ---------- clean ----------
clean:
	rm -rf ${BUILD}
