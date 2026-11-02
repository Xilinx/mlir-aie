//===- page_split_localbar.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// an oversized page must never split a local_barrier participant set
// across an .eop (G-localbar). The two local_barrier(0, 2) ops are participants
// of one barrier; the cost-based mid-split would fall between them (separating
// the group), so the splitter retargets to a legal cut after both participants.
// Both barriers therefore land on the SAME emitted page.

// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s

// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.local_barrier(0, 2)
// CHECK: aiex.cert.local_barrier(0, 2)
// CHECK: aiex.cert.page
// CHECK-NOT: aiex.cert.local_barrier

module {
  aie.device(npu2) {
    memref.global "private" constant @dataA : memref<1300xi32> = dense<0>
    memref.global "private" constant @dataB : memref<1300xi32> = dense<0>

    aiex.cert.uc_dma_chain @chainA {
      aiex.cert.uc_dma_bd @dataA, 0, 1300, false
    }
    aiex.cert.uc_dma_chain @chainB {
      aiex.cert.uc_dma_bd @dataB, 0, 1300, false
    }

    // A single oversized page. Cost accumulates past the mid-split target after
    // @chainA; the natural cut is before the second local_barrier, which would
    // separate the barrier group. The legal alternative cut is before @chainB,
    // keeping both local_barrier participants on the earlier page.
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.local_barrier(0, 2)
        aiex.cert.uc_dma_write_des_sync(@chainA)
        aiex.cert.local_barrier(0, 2)
        aiex.cert.uc_dma_write_des_sync(@chainB)
      }
    }
  }
}
