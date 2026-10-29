// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: split-file %s %t
// RUN: not aie-translate --aie-cert-to-asm %t/chain.mlir 2>&1 | FileCheck %s --check-prefix=CHAIN
// RUN: not aie-translate --aie-cert-to-asm %t/payload.mlir 2>&1 | FileCheck %s --check-prefix=PAYLOAD
// RUN: not aie-translate --aie-cert-to-asm %t/float.mlir 2>&1 | FileCheck %s --check-prefix=FLOAT
// RUN: not aie-translate --aie-cert-to-asm %t/group.mlir 2>&1 | FileCheck %s --check-prefix=GROUP
// RUN: not aie-translate --aie-cert-to-asm %t/patch.mlir 2>&1 | FileCheck %s --check-prefix=PAYLOAD

// CHAIN: uC DMA chain symbol not found: missing_chain
// PAYLOAD: Global symbol not found
// FLOAT: Global symbol initial value is not a dense int array
// GROUP: uC DMA chain symbol not found: missing_group_chain

//--- chain.mlir
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

//--- payload.mlir
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

//--- float.mlir
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

//--- group.mlir
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

//--- patch.mlir
module {
  aie.device(xcve3858) {
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.apply_offset_57(@missing_payload, 1, 0)
      }
    }
  }
}
