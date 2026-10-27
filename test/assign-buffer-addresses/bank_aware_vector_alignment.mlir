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

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | FileCheck %s

// The 144B `pad` ends at 1168, so the buffer behind it lands at 1184 (32 mod
// 64), where a 512-bit vector store is torn. It is bumped to 1216. Round-robin
// gives each of the four banks one buffer, so `b4` wraps back to bank 0 and
// takes that bumped address.
// CHECK-LABEL: module @bank_aware_needs_64B
// CHECK: aie.buffer @pad({{.*}}) {address = 1024 : i32, mem_bank = 0 : i32} : memref<72xbf16>
// CHECK: aie.buffer @b1({{.*}}) {address = 16384 : i32, mem_bank = 1 : i32} : memref<32xbf16>
// CHECK: aie.buffer @b2({{.*}}) {address = 32768 : i32, mem_bank = 2 : i32} : memref<32xbf16>
// CHECK: aie.buffer @b3({{.*}}) {address = 49152 : i32, mem_bank = 3 : i32} : memref<32xbf16>
// CHECK: aie.buffer @b4({{.*}}) {address = 1216 : i32, mem_bank = 0 : i32} : memref<32xbf16>
module @bank_aware_needs_64B {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @pad(%t) : memref<72xbf16>   // 144 B
    aie.buffer @b1(%t) : memref<32xbf16>
    aie.buffer @b2(%t) : memref<32xbf16>
    aie.buffer @b3(%t) : memref<32xbf16>
    aie.buffer @b4(%t) : memref<32xbf16>
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
// CHECK: aie.buffer @pad({{.*}}) {address = 1024 : i32, mem_bank = 0 : i32} : memref<72xbf16>
// CHECK: aie.buffer @vec({{.*}}) {address = 1216 : i32, mem_bank = 0 : i32} : memref<32xbf16>
module @bank_aware_membank_only_needs_64B {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @pad(%t) { mem_bank = 0 : i32 } : memref<72xbf16>   // 144 B
    aie.buffer @vec(%t) { mem_bank = 0 : i32 } : memref<32xbf16>   //  64 B
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
// CHECK: aie.buffer @rtp({{.*}}) {address = 55328 : i32{{.*}}} : memref<16xi32>
module @bank_aware_pinned_address_not_vetoed {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @rtp(%t) { address = 55328 : i32 } : memref<16xi32>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}
