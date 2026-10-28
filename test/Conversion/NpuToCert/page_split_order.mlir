//===- page_split_order.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// the page splitter preserves IR (textual) order within and across the
// result pages. An oversized job with two uc_dma_write_des_sync enqueues
// followed by a wait_tcts splits so each enqueue and the wait land on pages in
// their original order; splitting an enqueue from a later wait_tcts is a legal
// backward dependency.
//
// The first cut lands before wait_tcts, leaving @c1 + @c2 (an estimated 8084
// bytes) on the earlier page. That is still over the split trigger, so it is
// cut again between the two enqueues -- the second cut relies on the splitter
// being able to cut *before* the op that crosses the target when that op is its
// job's last (see page_split_last_op.mlir).

// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@c1)
// CHECK-NOT: aiex.cert.wait_tcts
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.uc_dma_write_des_sync(@c2)
// CHECK-NOT: aiex.cert.wait_tcts
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(3)
// CHECK: aiex.cert.wait_tcts(0, 6, 1)

module {
  aie.device(npu2) {
    memref.global "private" constant @c1data : memref<900xi32> = dense<0>
    memref.global "private" constant @c2data : memref<1100xi32> = dense<0>

    aiex.cert.uc_dma_chain @c1 {
      aiex.cert.uc_dma_bd @c1data, 0, 900, false
    }
    aiex.cert.uc_dma_chain @c2 {
      aiex.cert.uc_dma_bd @c2data, 0, 1100, false
    }

    // Oversized single job; the two chains do not merge (combined size >= page
    // size).
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des_sync(@c1)
        aiex.cert.uc_dma_write_des_sync(@c2)
        aiex.cert.wait_tcts(0, 6, 1)
      }
    }
  }
}
