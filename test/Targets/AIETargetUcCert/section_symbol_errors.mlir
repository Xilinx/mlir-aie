// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-translate --aie-cert-to-asm --split-input-file %s 2>&1 | FileCheck %s

// CHECK: uC DMA chain symbol not found: missing_chain
module {
  aie.device(xcve3858) {
    aiex.cert.section @missing_chain_section {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.uc_dma_write_des_sync(@missing_chain)
        }
      }
    }
  }
}

// -----

// CHECK: Global symbol not found
module {
  aie.device(xcve3858) {
    aiex.cert.uc_dma_chain @chain {
      aiex.cert.uc_dma_bd @missing_payload, 0x1000, 2, false
    }
    aiex.cert.section @missing_payload_section {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.uc_dma_write_des_sync(@chain)
        }
      }
    }
  }
}

// -----

// CHECK: Global symbol initial value is not a dense int array
module {
  aie.device(xcve3858) {
    memref.global "private" constant @float_payload : memref<2xf32> = dense<[1.0, 2.0]>
    aiex.cert.uc_dma_chain @chain {
      aiex.cert.uc_dma_bd @float_payload, 0x1000, 2, false
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des_sync(@chain)
      }
    }
  }
}

// -----

// CHECK: uC DMA chain symbol not found: missing_group_chain
module {
  aie.device(xcve3858) {
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.uc_dma_write_des_sync(@missing_group_chain)
        }
      }
    }
  }
}
