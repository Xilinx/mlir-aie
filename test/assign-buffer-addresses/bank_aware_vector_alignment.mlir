//===- bank_aware_vector_alignment.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Same requirement as vector_alignment.mlir, for the bank-aware allocator:
// core-tile buffers must be aligned to the widest vector load/store the core can
// issue (512b / 64B from AIE2P on), not to the 256b load/store bus.
//
// See AIETargetModel::getComputeTileMaxVectorAlignBits and aie_api's
// vector_ldst_align (aie_api/detail/ld_st.hpp).

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// The 144B `pad` ends at 1168, so the buffer behind it lands at 1184 (32 mod
// 64), where a 512-bit vector store is torn. It is bumped to 1216. The rest
// follow at 64B intervals, packed into one bank, because placement maximizes
// the free run it leaves behind.
// CHECK-LABEL: module @bank_aware_needs_64B
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "pad"} : memref<72xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1216 : i32, mem_bank = 0 : i32, sym_name = "b1"} : memref<32xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1280 : i32, mem_bank = 0 : i32, sym_name = "b2"} : memref<32xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1344 : i32, mem_bank = 0 : i32, sym_name = "b3"} : memref<32xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1408 : i32, mem_bank = 0 : i32, sym_name = "b4"} : memref<32xbf16>
module @bank_aware_needs_64B {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %pad = aie.buffer(%t) { sym_name = "pad" } : memref<72xbf16>   // 144 B
    %b1 = aie.buffer(%t) { sym_name = "b1" } : memref<32xbf16>
    %b2 = aie.buffer(%t) { sym_name = "b2" } : memref<32xbf16>
    %b3 = aie.buffer(%t) { sym_name = "b3" } : memref<32xbf16>
    %b4 = aie.buffer(%t) { sym_name = "b4" } : memref<32xbf16>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}

// -----

// Same as @bank_aware_needs_64B, but both buffers are pre-allocated with only
// a mem_bank attribute (no address): this goes through
// checkAndAddBufferWithMemBank rather than setBufferAddress. Without also
// applying the stricter alignment there, `vec` lands at 1184 (32 mod 64) and
// a 512-bit vector store to it is torn; it must be bumped to 1216.
// CHECK-LABEL: module @bank_aware_membank_only_needs_64B
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "pad"} : memref<72xbf16>
// CHECK: aie.buffer({{.*}}) {address = 1216 : i32, mem_bank = 0 : i32, sym_name = "vec"} : memref<32xbf16>
module @bank_aware_membank_only_needs_64B {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %pad = aie.buffer(%t) { sym_name = "pad", mem_bank = 0 : i32 } : memref<72xbf16>   // 144 B
    %vec = aie.buffer(%t) { sym_name = "vec", mem_bank = 0 : i32 } : memref<32xbf16>   //  64 B
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}

// -----

// An explicitly pinned address is the user's assertion and is often fixed by an
// external ABI (e.g. an RTP buffer a host writes at a known address). It is held
// only to the bus width, so pinning a 64B buffer at a 32-mod-64 address stays
// legal; only addresses this pass *chooses* get the stricter vector alignment.
// CHECK-LABEL: module @bank_aware_pinned_address_not_vetoed
// CHECK: aie.buffer({{.*}}) {address = 55328 : i32, {{.*}}sym_name = "rtp"} : memref<16xi32>
module @bank_aware_pinned_address_not_vetoed {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %rtp = aie.buffer(%t) { sym_name = "rtp", address = 55328 : i32 } : memref<16xi32>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}
