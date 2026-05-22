#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2026, Advanced Micro Devices, Inc.

# yolo26n-cls per-block + chain IRON build targets.
#
# Usage (with the mlir-aie env active — see ../../../utils/env_setup.sh):
#   make BLOCK=m0          -> build m0 .o + MLIR + xclbin + insts.bin
#   make run_ort BLOCK=m0  -> compile + run on NPU, bit-exact compare vs ORT
#   make chain             -> build full m0..m10 chain xclbin
#   make run_chain         -> run chain xclbin on NPU, bit-exact compare vs ORT
#   make clean             -> wipe build/
#
# data/ is regenerated from models/phase1_25k_xint8_acc0.8968.onnx by
# gen_yolo_data.py (gitignored).

SRCDIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD  := $(SRCDIR)/build
DATA   := $(SRCDIR)/data

BLOCK  ?= m0
ONNX   := $(SRCDIR)/models/phase1_25k_xint8_acc0.8968.onnx

# Peano flags — same shape as programming_examples/makefile-common's
# PEANOWRAP2P_FLAGS, inlined so the example is self-contained.
PEANOWRAP2P_FLAGS := -O2 -std=c++20 --target=aie2p-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw \
    -DNDEBUG -I $(MLIR_AIE_INSTALL_DIR)/include

# Kernel objects:
#   m0          -> yolo_m0_conv2dk3_silu_bias.o     (OIYX raw, in_c=3 padded)
#   m1          -> yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o (single-block use)
#   m2/m4       -> yolo_c3k2_small_*_m{2,4}.o     (per-block symbol mangling)
#   m3/m5/m7    -> yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_{m3,m5,m7}.o
#   m6         -> yolo_c3k2_heavy_*_m6.o + c3k2_small cv1_split/cv2_concat3
#   m8         -> 2-tile megakernel: yolo_m8_front_cv1_split_fused_m8.o,
#                  yolo_m8_back_cv3_cv2_fused_m8.o, plus streamed inner-pair
#                  kernels yolo_c3k2_heavy_inner_pair_cv{1,2}*_streamed_m8.o.
#   m9         -> single-block yolo_m9_*.o (no symbol mangling)
#   m10        -> single-block yolo_m10_*.o
# We build all .o files so any BLOCK or chain target can link without
# re-running the kernel build.
KERNEL_OBJS := $(BUILD)/yolo_m0_conv2dk3_silu_bias.o \
               $(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o

C3K2_SMALL_KERNELS := cv1_split m0_cv1 m0_cv2_skip cv2_concat3
C3K2_SMALL_BLOCKS  := m2 m4
C3K2_HEAVY_BLOCKS  := m6
C3K2_HEAVY_KERNELS := m_0_split inner_pair_cv1 inner_pair_cv2_skip cv3_concat2
C3K2_HEAVY_STREAMED_KERNELS := inner_pair_cv1_streamed inner_pair_cv2_skip_streamed
C3K2_SHARED_KERNELS := cv1_split cv2_concat3
C3K2_STREAMED_BLOCKS := m8
CHUNKED_CONV_BLOCKS  := m3 m5 m7

# c3k2_small kernels for m2/m4 (full set) and m6 (only the shared cv1_split,
# cv2_concat3 used by c3k2_heavy).
KERNEL_OBJS += $(foreach blk,$(C3K2_SMALL_BLOCKS), \
                 $(foreach k,$(C3K2_SMALL_KERNELS), \
                   $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o))
KERNEL_OBJS += $(foreach blk,$(C3K2_HEAVY_BLOCKS), \
                 $(foreach k,$(C3K2_SHARED_KERNELS), \
                   $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o))

# c3k2_heavy kernels for m6 (non-streamed; m8 uses fused-streamed instead).
KERNEL_OBJS += $(foreach blk,$(C3K2_HEAVY_BLOCKS), \
                 $(foreach k,$(C3K2_HEAVY_KERNELS), \
                   $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o))

# m8 megakernel: streamed inner-pair kernels (shared symbol for pair0/pair1
# since the megakernel calls the same kernel with different weights) plus
# the two fused kernels (front = cv1+m_0_split, back = cv3+cv2).
KERNEL_OBJS += $(foreach blk,$(C3K2_STREAMED_BLOCKS), \
                 $(foreach k,$(C3K2_HEAVY_STREAMED_KERNELS), \
                   $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o))
KERNEL_OBJS += $(foreach blk,$(C3K2_STREAMED_BLOCKS), \
                 $(BUILD)/yolo_m8_back_cv3_cv2_fused_$(blk).o \
                 $(BUILD)/yolo_m8_front_cv1_split_fused_$(blk).o)

# Chunked stride-2 conv (m3/m5/m7).
KERNEL_OBJS += $(foreach blk,$(CHUNKED_CONV_BLOCKS), \
                 $(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_$(blk).o)

M9_KERNELS := cv1_split qkv qkv_pack qk_pack qk_row attn_scale softmax_row \
              v_pack sv_row sv_row_acc pe_add_row proj_skip_row \
              ffn_0_silu_row ffn_1_skip_row cv2_concat2_streamed
KERNEL_OBJS += $(foreach k,$(M9_KERNELS),$(BUILD)/yolo_m9_$(k).o)

M10_KERNELS := conv2dk1_silu_xy_pool linear_gemm softmax
KERNEL_OBJS += $(foreach k,$(M10_KERNELS),$(BUILD)/yolo_m10_$(k).o)

MLIR   := $(BUILD)/aie_$(BLOCK).mlir
XCLBIN := $(BUILD)/final_$(BLOCK).xclbin
INSTS  := $(BUILD)/insts_$(BLOCK).bin

.PHONY: all run_ort chain run_chain time time_chain trace clean clean_chain data help FORCE

# Default goal: print help.
.DEFAULT_GOAL := help

# Number of samples per chain dispatch. Set to a value > 1 to build a chain
# xclbin that processes a batch in one launch (batched throughput mode).
# Must match between `make chain` and `make run_chain` / `make time_chain`;
# the .chain_n_samples stamp file below forces a chain rebuild if it changes.
CHAIN_N_SAMPLES ?= 1

# Extra args forwarded to scripts/time_block.py and scripts/time_chain.py,
# e.g. `make time BLOCK=m8 TIME_ARGS="--n-iters 50"`.
TIME_ARGS ?=

all: $(XCLBIN) $(INSTS)

define HELP_TEXT
yolo26n-cls on AMD Strix NPU — make targets

Per-block (set BLOCK=mN; default $(BLOCK)):
  make BLOCK=m8                 — build per-block xclbin + insts.bin
  make run_ort BLOCK=m8         — build + bit-exact compare vs ORT
  make time BLOCK=m8            — build + time on NPU (mean/min/max ms, fps)

Full m0..m10 chain:
  make chain                    — build chain xclbin + insts.bin (N=1)
  make run_chain                — build + bit-exact compare vs ORT (N=1)
  make time_chain               — build + time on NPU (N=1 latency)

Chain batched throughput (set CHAIN_N_SAMPLES=N at the make line):
  CHAIN_N_SAMPLES=15 make chain
  CHAIN_N_SAMPLES=15 make run_chain    — bit-exact, every sample
  CHAIN_N_SAMPLES=15 make time_chain   — per-sample ms, fps

Tracing (HW packet trace of the m8 megakernel; see scripts/trace_m8.sh):
  make trace                    — build with trace ops, run 1 iter, parse summary

Data + cleanup:
  make data                     — regenerate weights + LUTs from ONNX
  make clean_chain              — wipe just the chain xclbin/MLIR (cheap)
  make clean                    — wipe build/ entirely

Tuning the timing targets:
  make time BLOCK=m8 TIME_ARGS="--n-warmup 5 --n-iters 50"
endef
export HELP_TEXT

help:
	@echo "$$HELP_TEXT"

# Stamp file that tracks the CHAIN_N_SAMPLES value the chain xclbin was built
# with. Re-evaluated every `make` invocation (via FORCE prereq); only its
# mtime is bumped when the value changes, so the chain MLIR re-triggers
# without forcing unrelated targets to rebuild.
FORCE:
$(BUILD)/.chain_n_samples: FORCE
	@mkdir -p $(BUILD)
	@new="$(CHAIN_N_SAMPLES)"; old=$$(cat $@ 2>/dev/null || echo ""); \
	  if [ "$$old" != "$$new" ]; then \
	    echo "[chain] CHAIN_N_SAMPLES = $$new (was $${old:-unset}); chain MLIR will rebuild"; \
	    echo "$$new" > $@; \
	  fi

# Same stamp pattern for M9_STAGE — selects which staged build of m9 the
# per-block MLIR emits (1 = cv1 only, 10 = full block; default 10). Bumps
# the m9 MLIR's prereq when changed, so `M9_STAGE=1 make BLOCK=m9` and
# `M9_STAGE=10 make BLOCK=m9` rebuild correctly without `make clean`.
M9_STAGE ?= 10
$(BUILD)/.m9_stage: FORCE
	@mkdir -p $(BUILD)
	@new="$(M9_STAGE)"; old=$$(cat $@ 2>/dev/null || echo ""); \
	  if [ "$$old" != "$$new" ]; then \
	    echo "[m9] M9_STAGE = $$new (was $${old:-unset}); m9 MLIR will rebuild"; \
	    echo "$$new" > $@; \
	  fi

# Per-block ablation: pass `NOOP_BLOCK="m0 m1 ..."` (space-separated) to
# compile those blocks' kernels with -DNOOP_KERNEL. The kernel's extern "C"
# entry early-returns, preserving the chain's DMA/lock pattern while
# contributing ~0 ms of compute. Output is garbage; bit-exactness fails by
# design — used to attribute chain wall-clock to specific blocks.
NOOP_FLAGS_m0  := $(if $(filter m0, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m1  := $(if $(filter m1, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m2  := $(if $(filter m2, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m3  := $(if $(filter m3, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m4  := $(if $(filter m4, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m5  := $(if $(filter m5, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m6  := $(if $(filter m6, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m7  := $(if $(filter m7, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m8  := $(if $(filter m8, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m9  := $(if $(filter m9, $(NOOP_BLOCK)),-DNOOP_KERNEL,)
NOOP_FLAGS_m10 := $(if $(filter m10,$(NOOP_BLOCK)),-DNOOP_KERNEL,)

# Generic kernel .cc -> .o.
$(BUILD)/%.o: $(SRCDIR)/kernels/%.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -c $< -o $@

# m9/m10 use one set of kernels per block (no per-block symbol mangling).
# These pattern rules apply $(NOOP_FLAGS_m9) / _m10 to every m9_* / m10_*
# kernel, gated by NOOP_BLOCK from the env.
$(BUILD)/yolo_m9_%.o: $(SRCDIR)/kernels/yolo_m9_%.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) $(NOOP_FLAGS_m9) -c $< -o $@

$(BUILD)/yolo_m10_%.o: $(SRCDIR)/kernels/yolo_m10_%.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) $(NOOP_FLAGS_m10) -c $< -o $@

# Phase 1 vectorization: build the non-chunked conv2dk3_stride2 .o from
# the _vec.cc source instead of the scalar .cc. Same symbol name, drop-in.
# To revert (scalar): delete this rule + rebuild.
$(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o: \
    $(SRCDIR)/kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) $(NOOP_FLAGS_m1) -c $< -o $@

# Phase 3b: m0 stem vectorized (3x3 stride-2, OIYX raw, in_c=3 padded to 8).
$(BUILD)/yolo_m0_conv2dk3_silu_bias.o: \
    $(SRCDIR)/kernels/yolo_m0_conv2dk3_silu_bias_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) $(NOOP_FLAGS_m0) -c $< -o $@

# Phase 3c: m10 head conv+pool vectorized (1x1 256->1280 + GAP). Explicit
# rule overrides the yolo_m10_%.o pattern since the .cc base name differs
# from the .o (drops the _vec suffix).
$(BUILD)/yolo_m10_conv2dk1_silu_xy_pool.o: \
    $(SRCDIR)/kernels/yolo_m10_conv2dk1_silu_xy_pool_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) $(NOOP_FLAGS_m10) -c $< -o $@

# Phase 4: m9 cv1_split vectorized (chunked-OC 1x1 256->256 + SiLU + top/bot
# channel split). Drop-in: same symbol + ABI as the scalar .cc. Explicit rule
# overrides the yolo_m9_%.o pattern since the .cc base name differs from the
# .o (drops the _vec suffix).
$(BUILD)/yolo_m9_cv1_split.o: \
    $(SRCDIR)/kernels/yolo_m9_cv1_split_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) $(NOOP_FLAGS_m9) -c $< -o $@

# c3k2_heavy streamed inner-pair kernels (m8 megakernel: shared symbol
# for both pair0 and pair1 — they run on different tiles with different
# weights, but execute the same kernel code).
$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_heavy_inner_pair_cv1_streamed_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_inner_pair_cv1_streamed_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

# m8 megakernel: fused cv3+cv2 (m8_back). Computes cv3 once per row
# (on cv2 chunk 0), reuses cv3 output across the N_CV2_CHUNKS calls.
$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(eval $(BUILD)/yolo_m8_back_cv3_cv2_fused_$(blk).o: $(SRCDIR)/kernels/yolo_m8_back_cv3_cv2_fused_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

# m8 megakernel: fused cv1+m_0_split (m8_front). Accumulates bot across
# the cv1 chunk loop in a tile-local static; on last chunk runs m_0_split
# (both branches) producing split_a + split_b.
$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(eval $(BUILD)/yolo_m8_front_cv1_split_fused_$(blk).o: $(SRCDIR)/kernels/yolo_m8_front_cv1_split_fused_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

# c3k2_small kernels (m2/m4 + the cv1_split/cv2_concat3 shared with m6).
# -DKERNEL_SUFFIX=_mN gives each block a uniquely-named symbol.
$(foreach blk,$(C3K2_SMALL_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_m0_cv1_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_m0_cv1_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

$(foreach blk,$(C3K2_SMALL_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_m0_cv2_skip_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_m0_cv2_skip_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

$(foreach blk,$(C3K2_SMALL_BLOCKS) $(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_cv1_split_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_cv1_split_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

$(foreach blk,$(C3K2_SMALL_BLOCKS) $(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_cv2_concat3_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_cv2_concat3_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

# c3k2_heavy non-streamed kernels (m6 only).
$(foreach blk,$(C3K2_HEAVY_BLOCKS), \
  $(foreach k,$(C3K2_HEAVY_KERNELS), \
    $(eval $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_$(k)_vec.cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@)))

# Chunked stride-2 conv (m3/m5/m7). The kernel is compile-time shape-
# specialized via YOLO_IN_W/YOLO_IN_C/YOLO_OUT_C/YOLO_INTERIOR_MODE — these
# let peano fold addressing arithmetic into shifts and dead-strip the
# unused interior variant. INTERIOR_MODE=2 (2X×2OC, 4 accs) for blocks with
# enough x_tiles to amortize the pair-setup; INTERIOR_MODE=1 (OC×2 single-X,
# 2 accs) for m7 (x_tiles=4).
CHUNKED_SHAPE_m3 := -DYOLO_IN_W=128 -DYOLO_IN_C=64  -DYOLO_OUT_C=64  -DYOLO_INTERIOR_MODE=2
CHUNKED_SHAPE_m5 := -DYOLO_IN_W=64  -DYOLO_IN_C=128 -DYOLO_OUT_C=128 -DYOLO_INTERIOR_MODE=2
CHUNKED_SHAPE_m7 := -DYOLO_IN_W=32  -DYOLO_IN_C=128 -DYOLO_OUT_C=256 -DYOLO_INTERIOR_MODE=1

$(foreach blk,$(CHUNKED_CONV_BLOCKS), \
  $(eval $(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_$(blk).o: $(SRCDIR)/kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) $(CHUNKED_SHAPE_$(blk)) $(NOOP_FLAGS_$(blk)) -c $$< -o $$@))

SILU_LUTS := $(DATA)/model.0/silu_lut.bin $(DATA)/model.1/silu_lut.bin \
             $(DATA)/model.3/silu_lut.bin $(DATA)/model.5/silu_lut.bin \
             $(DATA)/model.7/silu_lut.bin \
             $(DATA)/model.9/cv1/silu_lut.bin \
             $(DATA)/model.9/m/m.0/ffn/ffn.0/silu_lut.bin \
             $(DATA)/model.9/cv2/silu_lut.bin \
             $(DATA)/model.10/silu_lut.bin

# IRON design -> MLIR. .m9_stage stamp forces rebuild when M9_STAGE changes
# (only relevant for BLOCK=m9 but the stamp is cheap and inert for others).
$(MLIR): $(SRCDIR)/aie2_yolo_per_block.py $(DATA)/manifest.json $(SILU_LUTS) \
         $(BUILD)/.m9_stage
	mkdir -p $(BUILD)
	cd $(SRCDIR) && M9_STAGE=$(M9_STAGE) python3 aie2_yolo_per_block.py $(BLOCK) > $@

# MLIR + kernel objs -> xclbin + insts.bin. aiecc runs from $(BUILD) so it
# resolves kernel .o by basename via the MLIR's `link_with` attr.
$(XCLBIN) $(INSTS): $(MLIR) $(KERNEL_OBJS)
	cd $(BUILD) && aiecc.py \
	    --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=$(notdir $(INSTS)) \
	    --xclbin-name=$(notdir $(XCLBIN)) $(notdir $(MLIR))

# Quark XINT8 ONNX -> data/* via gen_yolo_data.py.
$(DATA)/manifest.json: $(SRCDIR)/gen_yolo_data.py $(ONNX)
	cd $(SRCDIR) && python3 gen_yolo_data.py $(ONNX) > /dev/null

# Per-Conv SiLU LUTs. Single python invocation emits all bins.
$(SILU_LUTS): $(SRCDIR)/gen_yolo_silu_luts.py $(ONNX)
	cd $(SRCDIR) && python3 gen_yolo_silu_luts.py $(ONNX) > /dev/null

data: $(DATA)/manifest.json $(SILU_LUTS)

# Per-block bit-exact: build + run on NPU + compare vs ORT's
# /model.N/act post-SiLU tensor.
#   m0/m1/m2/m3/m4/m5/m6/m7 — straightforward per-block compare.
#   m9 (psa) — set M9_STAGE=1 to compare against cv1 post-SiLU, or
#              M9_STAGE=10 (default) to compare against cv2 post-SiLU
#              (the full m9 block output). Other stages emit intermediate
#              PSA tensors with no clean ORT analog; use `run_chain` for
#              end-to-end verification.
#   m10 — not yet wired; verify via `make run_chain`.
run_ort: $(XCLBIN) $(INSTS)
	cd $(SRCDIR) && M9_STAGE=$(M9_STAGE) python3 test_block_ort.py --block $(BLOCK) \
	    -x $(XCLBIN) -i $(INSTS) -k MLIR_AIE

# Per-block timing: warmup + N timed dispatches, reports mean/min/max ms + fps.
# Add `TIME_ARGS="--n-warmup 5 --n-iters 50"` to tune.
time: $(XCLBIN) $(INSTS)
	cd $(SRCDIR) && python3 scripts/time_block.py --block $(BLOCK) \
	    -x $(XCLBIN) -i $(INSTS) -k MLIR_AIE $(TIME_ARGS)

# --- Full m0..m10 chain ---
CHAIN_MLIR   := $(BUILD)/aie_chain.mlir
CHAIN_XCLBIN := $(BUILD)/final_chain.xclbin
CHAIN_INSTS  := $(BUILD)/insts_chain.bin

$(CHAIN_MLIR): $(SRCDIR)/aie2_yolo_iron_partial.py $(SRCDIR)/aie2_yolo_per_block.py \
               $(DATA)/manifest.json $(SILU_LUTS) $(BUILD)/.chain_n_samples
	mkdir -p $(BUILD)
	cd $(SRCDIR) && CHAIN_N_SAMPLES=$(CHAIN_N_SAMPLES) \
	    python3 aie2_yolo_iron_partial.py > $@

$(CHAIN_XCLBIN) $(CHAIN_INSTS): $(CHAIN_MLIR) $(KERNEL_OBJS)
	cd $(BUILD) && aiecc.py \
	    --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=$(notdir $(CHAIN_INSTS)) \
	    --xclbin-name=$(notdir $(CHAIN_XCLBIN)) $(notdir $(CHAIN_MLIR))

chain: $(CHAIN_XCLBIN) $(CHAIN_INSTS)

run_chain: $(CHAIN_XCLBIN) $(CHAIN_INSTS)
	cd $(SRCDIR) && CHAIN_N_SAMPLES=$(CHAIN_N_SAMPLES) \
	    python3 test_chain_ort.py -x $(CHAIN_XCLBIN) -i $(CHAIN_INSTS) -k MLIR_AIE

# Chain timing. For batched throughput, set CHAIN_N_SAMPLES at the make line.
time_chain: $(CHAIN_XCLBIN) $(CHAIN_INSTS)
	cd $(SRCDIR) && CHAIN_N_SAMPLES=$(CHAIN_N_SAMPLES) \
	    python3 scripts/time_chain.py -x $(CHAIN_XCLBIN) -i $(CHAIN_INSTS) \
	    -k MLIR_AIE $(TIME_ARGS)

# HW packet trace of the m8 megakernel. Sub-script handles the build/run/parse
# pipeline; honors TRACE_SIZE_PER_WORKER and TRACE_EVENTS env vars.
trace:
	bash $(SRCDIR)/scripts/trace_m8.sh

# Wipe just the chain artifacts (cheap; preserves kernel .o cache).
clean_chain:
	rm -rf $(CHAIN_MLIR) $(BUILD)/aie_chain.mlir.prj \
	       $(CHAIN_XCLBIN) $(CHAIN_INSTS) $(BUILD)/.chain_n_samples

clean:
	rm -rf $(BUILD)
