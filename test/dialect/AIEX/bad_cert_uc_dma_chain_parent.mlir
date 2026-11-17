//===- bad_cert_uc_dma_chain_parent.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// uC DMA chain names are device symbols, so a chain is defined at device
// level. A controller group that uses it names it from there.

aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<[1]>
  aiex.cert.attach_to_group(1) {
    // expected-error @+1 {{'aiex.cert.uc_dma_chain' op expects parent op 'aie.device'}}
    aiex.cert.uc_dma_chain @chain {
      aiex.cert.uc_dma_bd @data, 0x1000, 1, false
    }
  }
}

// -----

aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<[1]>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.attach_to_group(1) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des_sync(@chain)
      }
    }
  }
}
