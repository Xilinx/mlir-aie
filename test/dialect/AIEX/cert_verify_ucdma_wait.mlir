//===- cert_verify_ucdma_wait.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A cert.wait_uc_dma handle must be a local register r0..r7 written by an
// earlier cert.uc_dma_write_des in the same job.

// RUN: aie-opt --aie-cert-verify --split-input-file --verify-diagnostics %s

// Positive: write then wait in one job, including a handle reused by a second
// push.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(3, @chain)
      aiex.cert.wait_uc_dma(3)
      aiex.cert.uc_dma_write_des(3, @chain)
      aiex.cert.wait_uc_dma(3)
    }
  }
}

// -----

// The wait precedes the push.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      // expected-error@+1 {{cert.wait_uc_dma waits on r3, which no earlier cert.uc_dma_write_des in this job writes}}
      aiex.cert.wait_uc_dma(3)
      aiex.cert.uc_dma_write_des(3, @chain)
    }
  }
}

// -----

// The handle was written in another job; local registers do not carry over.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(3, @chain)
    }
    aiex.cert.job(2) {
      // expected-error@+1 {{cert.wait_uc_dma waits on r3, which no earlier cert.uc_dma_write_des in this job writes}}
      aiex.cert.wait_uc_dma(3)
    }
  }
}

// -----

// The wait names a different register from the push.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(3, @chain)
      // expected-error@+1 {{cert.wait_uc_dma waits on r4, which no earlier cert.uc_dma_write_des in this job writes}}
      aiex.cert.wait_uc_dma(4)
    }
  }
}

// -----

// A shared register cannot hold the handle in this dialect.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(8, @chain)
      // expected-error@+1 {{cert.wait_uc_dma handle r8 is a shared register}}
      aiex.cert.wait_uc_dma(8)
    }
  }
}
