// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Local registers r0..r7 belong to one job. Page splitting and full-page
// isolation rebuild jobs, so neither may separate a uC DMA handle from its
// wait. An asynchronous push also reads a patched payload, like a synchronous
// one.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// The cost-chosen cut falls inside job 2, between the push that writes r1 and
// the wait that reads it. That job is kept whole; the page is cut before it.
//
// CHECK-LABEL: aie.device(xcve3858) @no_cut_inside
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainA)
// CHECK-NOT: aiex.cert.uc_dma_write_des(
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des(1, @chainB)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainC)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.wait_uc_dma(1)
aie.device(xcve3858) @no_cut_inside {
  memref.global "private" constant @a : memref<900xi32> = dense<0>
  memref.global "private" constant @b : memref<900xi32> = dense<0>
  memref.global "private" constant @c : memref<300xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainA {
    aiex.cert.uc_dma_bd @a, 0, 900, false
  }
  aiex.cert.uc_dma_chain @chainB {
    aiex.cert.uc_dma_bd @b, 0, 900, false
  }
  aiex.cert.uc_dma_chain @chainC {
    aiex.cert.uc_dma_bd @c, 0, 300, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chainA)
    }
    aiex.cert.job(2) {
      aiex.cert.uc_dma_write_des(1, @chainB)
      aiex.cert.uc_dma_write_des_sync(@chainC)
      aiex.cert.wait_uc_dma(1)
    }
  }
}

// -----

// An asynchronous push in job 3 reads @qB, which job 2 patches, so the cut
// before job 3 is rejected and the page is cut before job 2.
//
// CHECK-LABEL: aie.device(xcve3858) @async_patched_payload
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chainQA)
// CHECK-NOT: aiex.cert.apply_offset_57(@qB
// CHECK: aiex.cert.page
// CHECK: aiex.cert.apply_offset_57(@qB
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des(1, @chainQB)
aie.device(xcve3858) @async_patched_payload {
  memref.global "private" constant @qA : memref<975xi32> = dense<0>
  memref.global "private" constant @qB : memref<1000xi32> = dense<0>
  aiex.cert.uc_dma_chain @chainQA {
    aiex.cert.uc_dma_bd @qA, 0, 975, false
  }
  aiex.cert.uc_dma_chain @chainQB {
    aiex.cert.uc_dma_bd @qB, 0, 1000, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.apply_offset_57(@qA, 1, 0)
      aiex.cert.uc_dma_write_des_sync(@chainQA)
    }
    aiex.cert.job(2) {
      aiex.cert.apply_offset_57(@qB, 1, 0)
    }
    aiex.cert.job(3) {
      aiex.cert.uc_dma_write_des(1, @chainQB)
      aiex.cert.wait_uc_dma(1)
    }
  }
}

// -----

// Isolation may move a load_pdi into its own job when the handle's push and
// wait both stay on one side of it.
//
// CHECK-LABEL: aie.device(xcve3858) @isolate_ok
// CHECK: aiex.cert.uc_dma_write_des(1, @chain)
// CHECK-NEXT: aiex.cert.wait_uc_dma(1)
// CHECK: aiex.cert.load_pdi(1, @cfg)
aie.device(xcve3858) @isolate_ok {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.section @cfg {
    aiex.cert.page {
      aiex.cert.job(10) {
        aiex.cert.write32(0x2000, 1)
      }
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(1, @chain)
      aiex.cert.wait_uc_dma(1)
      aiex.cert.load_pdi(1, @cfg)
    }
  }
}
