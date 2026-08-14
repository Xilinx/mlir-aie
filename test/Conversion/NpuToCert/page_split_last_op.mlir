//===- page_split_last_op.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The split-point search must not depend on where in a job the expensive op
// sits. estimateCost walks a job charging op costs and records the first
// position at which the running total crosses split_target; the natural
// candidate is "immediately after the op that crossed". When that op is the
// job's last real op, the only thing after it is the terminator, which is not a
// split point -- so the search has to fall back to cutting immediately *before*
// it. Without that fallback a page with its one fat op last was reported as
// offering no split point, while the very same ops with the fat one first split
// fine: a positional failure, not a size one.
//
// The first two cases below hold the same multiset of ops in opposite orders and
// estimate to the same 8484 bytes. Both must split.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// Case 1: fat op LAST. Running cost before it is 3660, under the 4000 target,
// and it alone carries the page to 8484, past the 8192 limit. Cut before it.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@last_small)
// CHECK-NOT: @last_fat
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@last_fat)
aie.device(npu2) {
  memref.global "private" constant @last_small_data : memref<900xi32> = dense<0>
  memref.global "private" constant @last_fat_data : memref<1200xi32> = dense<0>
  aiex.cert.uc_dma_chain @last_small {
    aiex.cert.uc_dma_bd @last_small_data, 0, 900, false
  }
  aiex.cert.uc_dma_chain @last_fat {
    aiex.cert.uc_dma_bd @last_fat_data, 0, 1200, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@last_small)
      aiex.cert.uc_dma_write_des_sync(@last_fat)
    }
  }
}

// -----

// Case 2: fat op FIRST -- the mirror image, which always split. Kept as the
// control, so the two orders are asserted to behave the same way.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@first_fat)
// CHECK-NOT: @first_small
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@first_small)
aie.device(npu2) {
  memref.global "private" constant @first_fat_data : memref<1200xi32> = dense<0>
  memref.global "private" constant @first_small_data : memref<900xi32> = dense<0>
  aiex.cert.uc_dma_chain @first_fat {
    aiex.cert.uc_dma_bd @first_fat_data, 0, 1200, false
  }
  aiex.cert.uc_dma_chain @first_small {
    aiex.cert.uc_dma_bd @first_small_data, 0, 900, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@first_fat)
      aiex.cert.uc_dma_write_des_sync(@first_small)
    }
  }
}

// -----

// Case 3: a distinct failure mode -- the crossing op is the last op of a job
// that is NOT the page's last job. The search does not stop at the end of that
// job, so it used to "recover" by taking the next position it could represent,
// after the following job's first op, and split there. That cut is on the wrong
// side of the fat op, so the earlier page came back over the limit and hit the
// same blind spot on the re-split: the miss was deferred, not avoided. The cut
// belongs before the fat op, on the first pass.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@mj_small)
// CHECK-NOT: @mj_fat
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@mj_fat)
// CHECK: aiex.cert.job(3)
// CHECK: aiex.cert.maskwrite32
// CHECK: aiex.cert.maskwrite32
aie.device(npu2) {
  memref.global "private" constant @mj_small_data : memref<900xi32> = dense<0>
  memref.global "private" constant @mj_fat_data : memref<1200xi32> = dense<0>
  aiex.cert.uc_dma_chain @mj_small {
    aiex.cert.uc_dma_bd @mj_small_data, 0, 900, false
  }
  aiex.cert.uc_dma_chain @mj_fat {
    aiex.cert.uc_dma_bd @mj_fat_data, 0, 1200, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@mj_small)
      aiex.cert.uc_dma_write_des_sync(@mj_fat)
    }
    aiex.cert.job(2) {
      aiex.cert.maskwrite32(6959104, 0, 1)
      aiex.cert.maskwrite32(6999568, 2, 2)
    }
  }
}

// -----

// Case 4: the fallback must not manufacture a degenerate split. The op that
// crosses the target is alone in its job, so there is nothing to cut before it
// in that job, and the cost search gives up -- only the exhaustive fallback
// resolves this page. Note that cutting before the crossing op whenever it is
// merely "not the job's first" also fires on the job terminator, which puts the
// whole job on the earlier page and an empty job on the later one, leaving the
// page unchanged so the greedy driver re-splits it forever. This case is what
// catches that.
//
// The fallback weighs every candidate by how close it lands to the target, and
// here the boundary between the two jobs (3684 bytes) beats the interior point
// between @alone_a and @alone_b (1860), so the cut goes between the jobs and
// each stays whole. That is the better of the two -- one page each way either
// way, but no job torn in half, which is also what lets @alone_a and @alone_b
// stay adjacent and merge into a single chain.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@alone_b)
// CHECK-NOT: @alone_fat
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@alone_fat)
aie.device(npu2) {
  memref.global "private" constant @alone_a_data : memref<450xi32> = dense<0>
  memref.global "private" constant @alone_b_data : memref<450xi32> = dense<0>
  memref.global "private" constant @alone_fat_data : memref<1200xi32> = dense<0>
  aiex.cert.uc_dma_chain @alone_a {
    aiex.cert.uc_dma_bd @alone_a_data, 0, 450, false
  }
  aiex.cert.uc_dma_chain @alone_b {
    aiex.cert.uc_dma_bd @alone_b_data, 0, 450, false
  }
  aiex.cert.uc_dma_chain @alone_fat {
    aiex.cert.uc_dma_bd @alone_fat_data, 0, 1200, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@alone_a)
      aiex.cert.uc_dma_write_des_sync(@alone_b)
    }
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des_sync(@alone_fat)
    }
  }
}
