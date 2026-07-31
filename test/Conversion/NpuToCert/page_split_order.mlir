//===- page_split_order.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// the page splitter preserves IR (textual) order within and across the
// two result pages. An oversized job with two uc_dma_write_des_sync enqueues
// followed by a wait_tcts splits so the enqueues stay on the earlier page and
// the wait_tcts lands on the later page (a legal backward dependency), with the
// internal order of each page preserved.

// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@c1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@c2)
// CHECK-NOT: aiex.cert.wait_tcts
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
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

    // Oversized single job. The cost-based cut lands before wait_tcts, so both
    // enqueues stay on the earlier page and the wait moves to the later page;
    // the two chains do not merge (combined size >= page size).
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des_sync(@c1)
        aiex.cert.uc_dma_write_des_sync(@c2)
        aiex.cert.wait_tcts(0, 6, 1)
      }
    }
  }
}
