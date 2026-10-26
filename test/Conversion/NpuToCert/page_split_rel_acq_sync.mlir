// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --cert-legalize-pages --split-input-file %s | FileCheck %s

// With 1980 payload words, the estimated page cost is 7996 bytes:
// 32 page overhead + 12 job overhead + 4 enqueue + 16 BD + 7920 data
// + 12 REL_ACQ_SYNC. It stays below the 8000-byte split trigger.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.rel_acq_sync(4096, 4100)
// CHECK-NOT: aiex.cert.page

aie.device(npu2) {
  memref.global "private" constant @payload : memref<1980xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @payload, 0, 1980, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chain)
      aiex.cert.rel_acq_sync(0x1000, 0x1004)
    }
  }
}

// -----

// One more payload word reaches exactly 8000 bytes. The 12-byte
// REL_ACQ_SYNC must trigger a split; counting it as 8 bytes or omitting it
// would leave the operations on one page.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.rel_acq_sync(4096, 4100)

aie.device(npu2) {
  memref.global "private" constant @payload : memref<1981xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @payload, 0, 1981, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chain)
      aiex.cert.rel_acq_sync(0x1000, 0x1004)
    }
  }
}
