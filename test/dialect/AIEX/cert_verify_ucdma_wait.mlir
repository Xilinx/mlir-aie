//===- cert_verify_ucdma_wait.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A cert.wait_uc_dma handle must be a local register r0..r7 whose last write
// earlier in the same job is a cert.uc_dma_write_des.

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

// -----

// A read into the handle's register between the push and the wait replaces
// the handle.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(3, @chain)
      aiex.cert.read32(3, 0x2000)
      // expected-error@+1 {{cert.wait_uc_dma waits on r3, but its last write is a 'aiex.cert.read32', which replaced the handle}}
      aiex.cert.wait_uc_dma(3)
    }
  }
}

// -----

// read32_d writes its value register, not its address register.
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.read32(4, 0x2000)
      aiex.cert.uc_dma_write_des(3, @chain)
      aiex.cert.read32_d(3, 5)
      aiex.cert.write32_d(3, 4) {address_is_reg, value_is_reg}
      aiex.cert.wait_uc_dma(3)
      aiex.cert.uc_dma_write_des(6, @chain)
      aiex.cert.read32_d(4, 6)
      // expected-error@+1 {{cert.wait_uc_dma waits on r6, but its last write is a 'aiex.cert.read32_d', which replaced the handle}}
      aiex.cert.wait_uc_dma(6)
    }
  }
}
