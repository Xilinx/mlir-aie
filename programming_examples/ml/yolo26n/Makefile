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
#   m2/m4       -> yolo_c3k2_small_*_m{2,4}.o      (per-block symbol mangling)
#   m3/m5/m7    -> yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_{m3,m5,m7}.o
#   m6/m8       -> yolo_c3k2_heavy_*_m{6,8}.o + c3k2_small cv1_split/cv2_concat3
#   m8          -> additional streamed cv1/cv2/inner_pair variants
#   m9          -> single-block yolo_m9_*.o (no symbol mangling)
#   m10         -> single-block yolo_m10_*.o
# We build all .o files so any BLOCK or chain target can link without
# re-running the kernel build.
KERNEL_OBJS := $(BUILD)/yolo_m0_conv2dk3_silu_bias.o \
               $(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o

C3K2_SMALL_KERNELS := cv1_split m0_cv1 m0_cv2_skip cv2_concat3
C3K2_SMALL_BLOCKS  := m2 m4
C3K2_HEAVY_BLOCKS  := m6 m8
C3K2_HEAVY_KERNELS := m_0_split inner_pair_cv1 inner_pair_cv2_skip cv3_concat2
C3K2_HEAVY_STREAMED_KERNELS := inner_pair_cv1_streamed inner_pair_cv2_skip_streamed
C3K2_SHARED_KERNELS := cv1_split cv2_concat3
C3K2_STREAMED_KERNELS := cv1_split_streamed cv2_concat3_streamed
C3K2_STREAMED_BLOCKS  := m8
CHUNKED_CONV_BLOCKS   := m3 m5 m7
C3K2_HEAVY_PAIR_IDX   := 0 1
C3K2_HEAVY_PAIR_NS_KERNELS := inner_pair_cv1 inner_pair_cv2_skip

KERNEL_OBJS += $(foreach blk,$(C3K2_SMALL_BLOCKS), \
                 $(foreach k,$(C3K2_SMALL_KERNELS), \
                   $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o))
KERNEL_OBJS += $(foreach blk,$(C3K2_HEAVY_BLOCKS), \
                 $(foreach k,$(C3K2_HEAVY_KERNELS), \
                   $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o))
KERNEL_OBJS += $(foreach blk,$(C3K2_STREAMED_BLOCKS), \
                 $(foreach k,$(C3K2_HEAVY_STREAMED_KERNELS), \
                   $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o))
# Per-PAIR variants of the (streamed and non-streamed) inner-pair kernels for
# m8. m8's c3k2_heavy has two inner pairs (pair0, pair1) each on its own tile;
# IRON requires distinct func symbols per tile, so suffix with _pairN_BLK.
KERNEL_OBJS += $(foreach blk,$(C3K2_STREAMED_BLOCKS), \
                 $(foreach k,$(C3K2_HEAVY_STREAMED_KERNELS), \
                   $(foreach p,$(C3K2_HEAVY_PAIR_IDX), \
                     $(BUILD)/yolo_c3k2_heavy_$(k)_pair$(p)_$(blk).o)))
KERNEL_OBJS += $(foreach blk,$(C3K2_STREAMED_BLOCKS), \
                 $(foreach k,$(C3K2_HEAVY_PAIR_NS_KERNELS), \
                   $(foreach p,$(C3K2_HEAVY_PAIR_IDX), \
                     $(BUILD)/yolo_c3k2_heavy_$(k)_pair$(p)_$(blk).o)))
KERNEL_OBJS += $(foreach blk,$(C3K2_HEAVY_BLOCKS), \
                 $(foreach k,$(C3K2_SHARED_KERNELS), \
                   $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o))
KERNEL_OBJS += $(foreach blk,$(C3K2_STREAMED_BLOCKS), \
                 $(foreach k,$(C3K2_STREAMED_KERNELS), \
                   $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o))
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

.PHONY: all run_ort chain run_chain clean data

all: $(XCLBIN) $(INSTS)

# Generic kernel .cc -> .o.
$(BUILD)/%.o: $(SRCDIR)/kernels/%.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -c $< -o $@

# Phase 1 vectorization: build the non-chunked conv2dk3_stride2 .o from
# the _vec.cc source instead of the scalar .cc. Same symbol name, drop-in.
# To revert (scalar): delete this rule + rebuild.
$(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o: \
    $(SRCDIR)/kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -c $< -o $@

# Phase 3b: m0 stem vectorized (3x3 stride-2, OIYX raw, in_c=3 padded to 8).
$(BUILD)/yolo_m0_conv2dk3_silu_bias.o: \
    $(SRCDIR)/kernels/yolo_m0_conv2dk3_silu_bias_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -c $< -o $@

# Phase 3c: m10 head conv+pool vectorized (1x1 256->1280 + GAP).
$(BUILD)/yolo_m10_conv2dk1_silu_xy_pool.o: \
    $(SRCDIR)/kernels/yolo_m10_conv2dk1_silu_xy_pool_vec.cc
	mkdir -p $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -c $< -o $@

# Phase 3d: c3k2_heavy inner_pair kernels (m6/m8) vectorized.
# inner_pair_cv1 = 3x3 stride-1, same math as c3k2_small m0_cv1.
# inner_pair_cv2_skip = 3x3 stride-1 + cross-scale skip-add.
$(foreach blk,$(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_heavy_inner_pair_cv1_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_inner_pair_cv1_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_heavy_inner_pair_cv2_skip_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_inner_pair_cv2_skip_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(foreach p,$(C3K2_HEAVY_PAIR_IDX), \
    $(eval $(BUILD)/yolo_c3k2_heavy_inner_pair_cv1_pair$(p)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_inner_pair_cv1_vec.cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_pair$(p)_$(blk) -c $$< -o $$@)))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(foreach p,$(C3K2_HEAVY_PAIR_IDX), \
    $(eval $(BUILD)/yolo_c3k2_heavy_inner_pair_cv2_skip_pair$(p)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_inner_pair_cv2_skip_vec.cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_pair$(p)_$(blk) -c $$< -o $$@)))

$(foreach blk,$(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_heavy_m_0_split_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_m_0_split_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_heavy_cv3_concat2_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_cv3_concat2_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

# Per-block c3k2_small kernels (m2/m4 + the cv1_split/cv2_concat3 sharing
# c3k2_heavy m6/m8 also build per-block variants of). -DKERNEL_SUFFIX=_mN
# gives each block a uniquely-named symbol.
$(foreach blk,$(C3K2_SMALL_BLOCKS) $(C3K2_HEAVY_BLOCKS), \
  $(foreach k,$(C3K2_SMALL_KERNELS), \
    $(eval $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_$(k).cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@)))

# Phase 3 vectorization overrides: build m0_cv1 + cv2_concat3 .o from _vec.cc.
# m2/m4 use the c3k2_small inner kernels directly; m6/m8 c3k2_heavy share the
# cv1_split + cv2_concat3 kernels (the heavy m0/inner_pair paths are separate).
$(foreach blk,$(C3K2_SMALL_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_m0_cv1_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_m0_cv1_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_SMALL_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_m0_cv2_skip_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_m0_cv2_skip_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_SMALL_BLOCKS) $(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_cv1_split_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_cv1_split_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_SMALL_BLOCKS) $(C3K2_HEAVY_BLOCKS), \
  $(eval $(BUILD)/yolo_c3k2_small_cv2_concat3_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_cv2_concat3_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_HEAVY_BLOCKS), \
  $(foreach k,$(C3K2_HEAVY_KERNELS), \
    $(eval $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_$(k).cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@)))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(foreach k,$(C3K2_HEAVY_STREAMED_KERNELS), \
    $(eval $(BUILD)/yolo_c3k2_heavy_$(k)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_$(k).cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@)))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(foreach k,$(C3K2_HEAVY_STREAMED_KERNELS), \
    $(foreach p,$(C3K2_HEAVY_PAIR_IDX), \
      $(eval $(BUILD)/yolo_c3k2_heavy_$(k)_pair$(p)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_$(k).cc ; \
        mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_pair$(p)_$(blk) -c $$< -o $$@))))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(foreach k,$(C3K2_HEAVY_PAIR_NS_KERNELS), \
    $(foreach p,$(C3K2_HEAVY_PAIR_IDX), \
      $(eval $(BUILD)/yolo_c3k2_heavy_$(k)_pair$(p)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_heavy_$(k).cc ; \
        mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_pair$(p)_$(blk) -c $$< -o $$@))))

$(foreach blk,$(CHUNKED_CONV_BLOCKS), \
  $(eval $(BUILD)/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_$(blk).o: $(SRCDIR)/kernels/yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc ; \
    mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@))

$(foreach blk,$(C3K2_STREAMED_BLOCKS), \
  $(foreach k,$(C3K2_STREAMED_KERNELS), \
    $(eval $(BUILD)/yolo_c3k2_small_$(k)_$(blk).o: $(SRCDIR)/kernels/yolo_c3k2_small_$(k).cc ; \
      mkdir -p $(BUILD) && $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -DKERNEL_SUFFIX=_$(blk) -c $$< -o $$@)))

SILU_LUTS := $(DATA)/model.0/silu_lut.bin $(DATA)/model.1/silu_lut.bin \
             $(DATA)/model.3/silu_lut.bin $(DATA)/model.5/silu_lut.bin \
             $(DATA)/model.7/silu_lut.bin \
             $(DATA)/model.9/cv1/silu_lut.bin \
             $(DATA)/model.9/m/m.0/ffn/ffn.0/silu_lut.bin \
             $(DATA)/model.9/cv2/silu_lut.bin \
             $(DATA)/model.10/silu_lut.bin

# IRON design -> MLIR.
$(MLIR): $(SRCDIR)/aie2_yolo_per_block.py $(DATA)/manifest.json $(SILU_LUTS)
	mkdir -p $(BUILD)
	cd $(SRCDIR) && python3 aie2_yolo_per_block.py $(BLOCK) > $@

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

# Run on NPU and bit-exact compare vs ORT's /model.N/act post-SiLU tensor.
# Works for any per-block target; the block is inferred from $(BLOCK).
run_ort: $(XCLBIN) $(INSTS)
	cd $(SRCDIR) && python3 test_block_ort.py --block $(BLOCK) \
	    -x $(XCLBIN) -i $(INSTS) -k MLIR_AIE

# --- Full m0..m10 chain ---
CHAIN_MLIR   := $(BUILD)/aie_chain.mlir
CHAIN_XCLBIN := $(BUILD)/final_chain.xclbin
CHAIN_INSTS  := $(BUILD)/insts_chain.bin

$(CHAIN_MLIR): $(SRCDIR)/aie2_yolo_iron_partial.py $(SRCDIR)/aie2_yolo_per_block.py \
               $(DATA)/manifest.json $(SILU_LUTS)
	mkdir -p $(BUILD)
	cd $(SRCDIR) && python3 aie2_yolo_iron_partial.py > $@

$(CHAIN_XCLBIN) $(CHAIN_INSTS): $(CHAIN_MLIR) $(KERNEL_OBJS)
	cd $(BUILD) && aiecc.py \
	    --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
	    --aie-generate-npu-insts --npu-insts-name=$(notdir $(CHAIN_INSTS)) \
	    --xclbin-name=$(notdir $(CHAIN_XCLBIN)) $(notdir $(CHAIN_MLIR))

chain: $(CHAIN_XCLBIN) $(CHAIN_INSTS)

run_chain: $(CHAIN_XCLBIN) $(CHAIN_INSTS)
	cd $(SRCDIR) && python3 test_chain_ort.py \
	    -x $(CHAIN_XCLBIN) -i $(CHAIN_INSTS) -k MLIR_AIE

clean:
	rm -rf $(BUILD)
