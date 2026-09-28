// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --cert-legalize-pages %s | FileCheck %s

// The data, DMA instruction/BD, page header, and final write cost 7996 bytes.
// The 16-byte maskpoll takes the page past the 8000-byte splitting threshold.
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(1)
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.job(2)
// CHECK: aiex.cert.maskpoll32(119328, 16777216, 0)
// CHECK-NEXT: aiex.cert.write32(119316, 1)
// CHECK-NOT: aiex.cert.page

aie.device(npu2) {
  memref.global "private" constant @data : memref<1980xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 131072, 1980, false
  }
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@chain)
    aiex.cert.maskpoll32(119328, 16777216, 0)
    aiex.cert.write32(119316, 1)
  }
}
