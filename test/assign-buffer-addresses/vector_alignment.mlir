//===- vector_alignment.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Core-tile buffers must be aligned to the widest vector load/store the core
// can issue, which from AIE2P on is 512 bits (64B) -- stricter than the 256-bit
// load/store bus. A buffer only 32B-aligned that is then written by a full
// 512-bit vector store silently loses the half of the store past the 64B line.
//
// See AIETargetModel::getComputeTileMaxVectorAlignBits and aie_api's
// vector_ldst_align (aie_api/detail/ld_st.hpp).

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s

// AIE2P (npu2): `pad` is 144B, so without the fix the following 64B buffer
// lands at 1184 (32-aligned but 32 mod 64) and a 512-bit store to it is torn.
// It must be bumped to 1216.
// CHECK-LABEL: module @aie2p_core_needs_64B
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "pad"} : memref<72xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1216 : i32, sym_name = "vec"} : memref<32xbf16>
module @aie2p_core_needs_64B {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %pad = aie.buffer(%t) { sym_name = "pad" } : memref<72xbf16>   // 144 B
    %vec = aie.buffer(%t) { sym_name = "vec" } : memref<32xbf16>   //  64 B
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}

// -----

// Buffers too small to hold a full-width vector cannot be accessed by one
// without going out of bounds, so they keep the cheaper 32B bus alignment and
// cost no extra padding.
// CHECK-LABEL: module @aie2p_small_buffers_unpadded
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "s0"} : memref<8xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1056 : i32, sym_name = "s1"} : memref<8xbf16>
module @aie2p_small_buffers_unpadded {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %s0 = aie.buffer(%t) { sym_name = "s0" } : memref<8xbf16>      // 16 B
    %s1 = aie.buffer(%t) { sym_name = "s1" } : memref<8xbf16>      // 16 B
    aie.core(%t) {
      aie.end
    } { stackSize = 1024 : i32 }
  }
}

// -----

// AIE2 (npu1) keeps 32B alignment: its widest vector access is 256 bits, so
// this change must not perturb existing AIE2 layouts.
// CHECK-LABEL: module @aie2_unchanged
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, sym_name = "pad"} : memref<72xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1184 : i32, sym_name = "vec"} : memref<32xbf16>
module @aie2_unchanged {
  aie.device(npu1) {
    %t = aie.tile(0, 2)
    %pad = aie.buffer(%t) { sym_name = "pad" } : memref<72xbf16>
    %vec = aie.buffer(%t) { sym_name = "vec" } : memref<32xbf16>
    aie.core(%t) {
      aie.end
    } { stackSize = 1024 : i32 }
  }
}

// -----

// MemTile buffers are reached by DMA, not by core vector load/stores, so they
// keep the 4B DMA alignment and gain no padding.
// CHECK-LABEL: module @aie2p_memtile_unchanged
// CHECK: aie.buffer({{.*}}) {address = 0 : i32, sym_name = "m0"} : memref<72xbf16>
// CHECK: aie.buffer({{.*}}) {address = 144 : i32, sym_name = "m1"} : memref<32xbf16>
module @aie2p_memtile_unchanged {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %m0 = aie.buffer(%t) { sym_name = "m0" } : memref<72xbf16>
    %m1 = aie.buffer(%t) { sym_name = "m1" } : memref<32xbf16>
  }
}

// -----

// https://github.com/Xilinx/mlir-aie/issues/2579: buffers are placed after the
// stack, so a stack_size that is not a multiple of 64 used to leave every
// following buffer 32-aligned, silently breaking aie::load_v / aie::store_v.
// The alignment must not depend on the stack size.
// CHECK-LABEL: module @aie2p_unaligned_stack_size
// CHECK: aie.buffer({{.*}}) {address = 1088 : i32, sym_name = "bufin"} : memref<32xi32>
// CHECK: aie.buffer({{.*}}) {address = 1216 : i32, sym_name = "bufout"} : memref<32xi32>
module @aie2p_unaligned_stack_size {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %in = aie.buffer(%t) { sym_name = "bufin" } : memref<32xi32>
    %out = aie.buffer(%t) { sym_name = "bufout" } : memref<32xi32>
    aie.core(%t) {
      aie.end
    } { stack_size = 1028 : i32 }
  }
}

// -----

// An explicitly pinned address is the user's assertion and is often fixed by an
// external ABI (e.g. an RTP buffer a host writes at a known address). It is held
// only to the bus width, so pinning a 64B buffer at a 32-mod-64 address stays
// legal; only addresses this pass *chooses* get the stricter vector alignment.
// CHECK-LABEL: module @aie2p_pinned_address_not_vetoed
// CHECK: aie.buffer({{.*}}) {address = 55328 : i32, sym_name = "rtp"} : memref<16xi32>
module @aie2p_pinned_address_not_vetoed {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %rtp = aie.buffer(%t) { sym_name = "rtp", address = 55328 : i32 } : memref<16xi32>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}
