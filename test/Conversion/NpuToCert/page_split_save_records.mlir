// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Page text costs of the debug record ops: SAVE_REGISTER is 12 bytes and
// SAVE_TIMESTAMPS 8.

// RUN: aie-opt --cert-legalize-pages --split-input-file %s | FileCheck %s

// With 1980 payload words the page costs 32 page overhead + 16 enqueue + 16 BD
// + 7920 data + 12 SAVE_REGISTER = 7996 bytes, below the 8000-byte split
// trigger.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.save_register
// CHECK-NOT: aiex.cert.page
aie.device(xcve3858) {
  memref.global "private" constant @payload : memref<1980xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @payload, 0, 1980, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chain)
      aiex.cert.save_register(0x1000, 9)
    }
  }
}

// -----

// One more payload word reaches 8000 bytes and must split; counting
// SAVE_REGISTER as 8 bytes would not.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.save_register
aie.device(xcve3858) {
  memref.global "private" constant @payload : memref<1981xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @payload, 0, 1981, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chain)
      aiex.cert.save_register(0x1000, 9)
    }
  }
}

// -----

// With 1981 payload words and an 8-byte SAVE_TIMESTAMPS the page costs 7996
// bytes and stays whole.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK-NOT: aiex.cert.page
// CHECK: aiex.cert.save_timestamps
// CHECK-NOT: aiex.cert.page
aie.device(xcve3858) {
  memref.global "private" constant @payload : memref<1981xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @payload, 0, 1981, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chain)
      aiex.cert.save_timestamps(7)
    }
  }
}

// -----

// One more payload word reaches 8000 bytes and must split; counting
// SAVE_TIMESTAMPS as 4 bytes would not.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
// CHECK: aiex.cert.page
// CHECK: aiex.cert.save_timestamps
aie.device(xcve3858) {
  memref.global "private" constant @payload : memref<1982xi32> = dense<0>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @payload, 0, 1982, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@chain)
      aiex.cert.save_timestamps(7)
    }
  }
}
