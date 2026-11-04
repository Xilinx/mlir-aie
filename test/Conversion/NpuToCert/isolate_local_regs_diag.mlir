// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Isolating a load_pdi into its own job would leave the wait in a different
// job from the push that wrote its handle, so the pass fails.

// RUN: aie-opt -cert-legalize-pages --verify-diagnostics %s

aie.device(xcve3858) {
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
    // expected-error@+1 {{job 1 must be split around its aiex.cert.load_pdi, but local register r1 is written before it and read after it}}
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(1, @chain)
      aiex.cert.load_pdi(1, @cfg)
      aiex.cert.wait_uc_dma(1)
    }
  }
}
