//===- page_split_job_boundary.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A page can be oversized without any single job being oversized: several small
// jobs whose sizes merely sum past the limit. The cut such a page wants is a
// job boundary -- whole jobs on either side, nothing partitioned -- and the
// split-point search has to be able to name one. Both searches used to consider
// only positions with an op of the *same* job ahead of them, so a job's first op
// was never a candidate and these pages were reported as offering no split point
// at all. A position is legal as soon as anything on the page precedes it, in
// the same job or an earlier one.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// Case 1: the minimal shape -- two one-op jobs, 9696 bytes together, neither
// with an interior point to cut at. One job per page, both kept whole, and
// because no job is partitioned the job ids are untouched.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@jb_c1)
// CHECK-NOT: @jb_c2
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@jb_c2)
aie.device(npu2) {
  memref.global "private" constant @jb_d1 : memref<1200xi32> = dense<0>
  memref.global "private" constant @jb_d2 : memref<1200xi32> = dense<1>
  aiex.cert.uc_dma_chain @jb_c1 {
    aiex.cert.uc_dma_bd @jb_d1, 0, 1200, false
  }
  aiex.cert.uc_dma_chain @jb_c2 {
    aiex.cert.uc_dma_bd @jb_d2, 0, 1200, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@jb_c1)
    }
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des_sync(@jb_c2)
    }
  }
}

// -----

// Case 2: four one-op jobs of 2432 bytes each, 9760 in total. Several boundaries
// are legal, so the search picks the one closest to the 4096 target -- the
// middle one, two jobs each way -- rather than the first it happens to see.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@sum_c1)
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@sum_c2)
// CHECK-NOT: @sum_c3
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(3)
// CHECK: aiex.cert.uc_dma_write_des_sync(@sum_c3)
// CHECK: aiex.cert.job(4)
// CHECK: aiex.cert.uc_dma_write_des_sync(@sum_c4)
aie.device(npu2) {
  memref.global "private" constant @sum_d1 : memref<600xi32> = dense<0>
  memref.global "private" constant @sum_d2 : memref<600xi32> = dense<1>
  memref.global "private" constant @sum_d3 : memref<600xi32> = dense<2>
  memref.global "private" constant @sum_d4 : memref<600xi32> = dense<3>
  aiex.cert.uc_dma_chain @sum_c1 {
    aiex.cert.uc_dma_bd @sum_d1, 0, 600, false
  }
  aiex.cert.uc_dma_chain @sum_c2 {
    aiex.cert.uc_dma_bd @sum_d2, 0, 600, false
  }
  aiex.cert.uc_dma_chain @sum_c3 {
    aiex.cert.uc_dma_bd @sum_d3, 0, 600, false
  }
  aiex.cert.uc_dma_chain @sum_c4 {
    aiex.cert.uc_dma_bd @sum_d4, 0, 600, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@sum_c1)
    }
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des_sync(@sum_c2)
    }
    aiex.cert.job(3) {
      aiex.cert.uc_dma_write_des_sync(@sum_c3)
    }
    aiex.cert.job(4) {
      aiex.cert.uc_dma_write_des_sync(@sum_c4)
    }
  }
}

// -----

// Case 3: the same gap reached a step later. The one interior cut on this page
// (in the first job, before its fat op) is the one the cost search takes, and
// the page it leaves behind -- one fat op per one-op job -- is the shape of case
// 1 again. Without a boundary cut available the second pass has nothing left to
// try and the page fails, with the diagnostic pointing at a page that no longer
// exists in the input. Three pages and no diagnostic is the whole assertion.

// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@def_tiny)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@def_c1)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@def_c2)
aie.device(npu2) {
  memref.global "private" constant @def_dt : memref<50xi32> = dense<0>
  memref.global "private" constant @def_d1 : memref<1200xi32> = dense<0>
  memref.global "private" constant @def_d2 : memref<1200xi32> = dense<1>
  aiex.cert.uc_dma_chain @def_tiny {
    aiex.cert.uc_dma_bd @def_dt, 0, 50, false
  }
  aiex.cert.uc_dma_chain @def_c1 {
    aiex.cert.uc_dma_bd @def_d1, 0, 1200, false
  }
  aiex.cert.uc_dma_chain @def_c2 {
    aiex.cert.uc_dma_bd @def_d2, 0, 1200, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@def_tiny)
      aiex.cert.uc_dma_write_des_sync(@def_c1)
    }
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des_sync(@def_c2)
    }
  }
}

// -----

// Case 4: a boundary cut is still subject to local_barrier co-location
// (G-localbar), and here it is the only cut that obeys it. The local_barrier
// group spans jobs 2 and 3, so every interior
// position inside them separates its participants; the boundary between job 1
// and job 2 sits just ahead of the group and keeps it whole. Cutting there is
// the difference between splitting this page and refusing to (the cost search's
// own pick, before job 2's fat op, splits the group).

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@bar_ca)
// CHECK-NOT: aiex.cert.local_barrier
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.local_barrier(0, 2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@bar_cb)
// CHECK: aiex.cert.job(3)
// CHECK: aiex.cert.uc_dma_write_des_sync(@bar_cc)
// CHECK: aiex.cert.local_barrier(0, 2)
aie.device(npu2) {
  memref.global "private" constant @bar_da : memref<800xi32> = dense<0>
  memref.global "private" constant @bar_db : memref<800xi32> = dense<1>
  memref.global "private" constant @bar_dc : memref<800xi32> = dense<2>
  aiex.cert.uc_dma_chain @bar_ca {
    aiex.cert.uc_dma_bd @bar_da, 0, 800, false
  }
  aiex.cert.uc_dma_chain @bar_cb {
    aiex.cert.uc_dma_bd @bar_db, 0, 800, false
  }
  aiex.cert.uc_dma_chain @bar_cc {
    aiex.cert.uc_dma_bd @bar_dc, 0, 800, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@bar_ca)
    }
    aiex.cert.job(2) {
      aiex.cert.local_barrier(0, 2)
      aiex.cert.uc_dma_write_des_sync(@bar_cb)
    }
    aiex.cert.job(3) {
      aiex.cert.uc_dma_write_des_sync(@bar_cc)
      aiex.cert.local_barrier(0, 2)
    }
  }
}
