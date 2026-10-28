//===- merge_dma_chains_ordering.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Limits on MergeConsecutiveCertUcDmaWriteDesSyncOps. Fusing two enqueues moves
// the earlier DMA to the later enqueue's position, so the merge is legal only
// when nothing in between observes the DMA and when neither chain symbol is
// shared with an enqueue outside the pair.
//
// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// A local_barrier between the two enqueues blocks the merge: the first DMA must
// stay ahead of the barrier.
// CHECK-LABEL: @d1_barrier
// CHECK:      aiex.cert.uc_dma_chain @c1_barrier {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1_barrier, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.uc_dma_chain @c2_barrier {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d2_barrier, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_barrier)
// CHECK-NEXT:   aiex.cert.local_barrier(0, 2)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c2_barrier)
// CHECK-NEXT: }
aie.device(npu2) {
  memref.global "private" constant @d1_barrier : memref<8xi32> = dense<0>
  memref.global "private" constant @d2_barrier : memref<8xi32> = dense<1>
  aiex.cert.uc_dma_chain @c1_barrier { aiex.cert.uc_dma_bd @d1_barrier, 0, 8, false }
  aiex.cert.uc_dma_chain @c2_barrier { aiex.cert.uc_dma_bd @d2_barrier, 0, 8, false }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1_barrier)
      aiex.cert.local_barrier(0, 2)
      aiex.cert.uc_dma_write_des_sync(@c2_barrier)
    }
  }
}

// -----

// A remote_barrier also blocks the merge.
// CHECK-LABEL: @d1_remote
// CHECK:      aiex.cert.uc_dma_chain @c1_remote {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1_remote, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.uc_dma_chain @c2_remote {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d2_remote, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_remote)
// CHECK-NEXT:   aiex.cert.remote_barrier(1, 3)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c2_remote)
// CHECK-NEXT: }
aie.device(npu2) {
  memref.global "private" constant @d1_remote : memref<8xi32> = dense<0>
  memref.global "private" constant @d2_remote : memref<8xi32> = dense<1>
  aiex.cert.uc_dma_chain @c1_remote { aiex.cert.uc_dma_bd @d1_remote, 0, 8, false }
  aiex.cert.uc_dma_chain @c2_remote { aiex.cert.uc_dma_bd @d2_remote, 0, 8, false }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1_remote)
      aiex.cert.remote_barrier(1, 3)
      aiex.cert.uc_dma_write_des_sync(@c2_remote)
    }
  }
}

// -----

// cert.nop has no architectural effect, so it stays transparent to the merge.
// CHECK-LABEL: @d1_nop
// CHECK:      aiex.cert.uc_dma_chain @[[SYM:[a-zA-Z0-9_]+]] {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1_nop, 0, 8, true
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d2_nop, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.nop
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@[[SYM]])
// CHECK-NEXT: }
aie.device(npu2) {
  memref.global "private" constant @d1_nop : memref<8xi32> = dense<0>
  memref.global "private" constant @d2_nop : memref<8xi32> = dense<1>
  aiex.cert.uc_dma_chain @c1_nop { aiex.cert.uc_dma_bd @d1_nop, 0, 8, false }
  aiex.cert.uc_dma_chain @c2_nop { aiex.cert.uc_dma_bd @d2_nop, 0, 8, false }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1_nop)
      aiex.cert.nop
      aiex.cert.uc_dma_write_des_sync(@c2_nop)
    }
  }
}

// -----

// @c1_shared is enqueued again from another job, so it may be neither folded
// away nor erased -- erasing it would leave job(2) pointing at a dead symbol.
// CHECK-LABEL: @d1_shared
// CHECK:      aiex.cert.uc_dma_chain @c1_shared {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1_shared, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.uc_dma_chain @c2_shared {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d2_shared, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_shared)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c2_shared)
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(2) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_shared)
// CHECK-NEXT: }
aie.device(npu2) {
  memref.global "private" constant @d1_shared : memref<8xi32> = dense<0>
  memref.global "private" constant @d2_shared : memref<8xi32> = dense<1>
  aiex.cert.uc_dma_chain @c1_shared { aiex.cert.uc_dma_bd @d1_shared, 0, 8, false }
  aiex.cert.uc_dma_chain @c2_shared { aiex.cert.uc_dma_bd @d2_shared, 0, 8, false }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1_shared)
      aiex.cert.uc_dma_write_des_sync(@c2_shared)
    }
  }
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des_sync(@c1_shared)
    }
  }
}

// -----

// Mirror image of @d1_shared: this time it's the *surviving* (later) chain
// that's shared with another job, not the erased (earlier) one. Folding
// @c1_sur's BD into @c2_sur would still be wrong -- job(2)'s enqueue of
// @c2_sur would silently pick up @c1_sur's transfer too.
// CHECK-LABEL: @d1_sur
// CHECK:      aiex.cert.uc_dma_chain @c1_sur {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1_sur, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.uc_dma_chain @c2_sur {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d2_sur, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_sur)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c2_sur)
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(2) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c2_sur)
// CHECK-NEXT: }
aie.device(npu2) {
  memref.global "private" constant @d1_sur : memref<8xi32> = dense<0>
  memref.global "private" constant @d2_sur : memref<8xi32> = dense<1>
  aiex.cert.uc_dma_chain @c1_sur { aiex.cert.uc_dma_bd @d1_sur, 0, 8, false }
  aiex.cert.uc_dma_chain @c2_sur { aiex.cert.uc_dma_bd @d2_sur, 0, 8, false }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1_sur)
      aiex.cert.uc_dma_write_des_sync(@c2_sur)
    }
  }
  aiex.cert.page {
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des_sync(@c2_sur)
    }
  }
}

// -----

// A chain enqueued twice within one job is likewise left alone: folding into it
// would change what the second enqueue transfers.
// CHECK-LABEL: @d1_self
// CHECK:      aiex.cert.uc_dma_chain @c1_self {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1_self, 0, 8, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_self)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c1_self)
// CHECK-NEXT: }
aie.device(npu2) {
  memref.global "private" constant @d1_self : memref<8xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1_self { aiex.cert.uc_dma_bd @d1_self, 0, 8, false }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1_self)
      aiex.cert.uc_dma_write_des_sync(@c1_self)
    }
  }
}
