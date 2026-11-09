// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each controller group resolves labels in its own block, so every group that
// uses a device-level chain or patched payload needs its own copy.
// RUN: aie-translate --aie-cert-to-asm --split-input-file %s | FileCheck %s
// RUN: aie-opt --cert-legalize-pages --split-input-file %s | aie-translate --aie-cert-to-asm --split-input-file | FileCheck %s --check-prefix=MERGE

// CHECK-LABEL: .attach_to_group 0
// CHECK:       START_JOB 0
// CHECK:       EOF
// CHECK-NOT:   shared:
// CHECK-NOT:   other:
// CHECK-NOT:   data:
// CHECK-LABEL: .attach_to_group 1
// CHECK:         uC_DMA_WRITE_DES_SYNC  @shared
// CHECK:         uC_DMA_WRITE_DES_SYNC  @shared
// CHECK-NEXT:    uC_DMA_WRITE_DES_SYNC  @other
// CHECK:       EOF
// CHECK:       shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x00001000, @data, 1, 0, 0
// CHECK-NOT:   shared:
// CHECK:       other:
// CHECK-NEXT:    UC_DMA_BD       0, 0x00001004, @data, 1, 0, 0
// CHECK:       data:
// CHECK-NEXT:    .long           0x00000001
// CHECK-NOT:   data:
// CHECK-NOT:   other:
// CHECK-LABEL: .attach_to_group 2
// CHECK:         uC_DMA_WRITE_DES_SYNC  @shared
// CHECK:       EOF
// CHECK:       shared:
// CHECK-NEXT:    UC_DMA_BD       0, 0x00001000, @data, 1, 0, 0
// CHECK:       data:
// CHECK-NEXT:    .long           0x00000001
// CHECK-NOT:   other:

module {
  aie.device(xcve3858) {
    memref.global "private" constant @data : memref<1xi32> = dense<[1]>
    aiex.cert.uc_dma_chain @shared {
      aiex.cert.uc_dma_bd @data, 0x1000, 1, false
    }
    aiex.cert.uc_dma_chain @other {
      aiex.cert.uc_dma_bd @data, 0x1004, 1, false
    }
    aiex.cert.page {
      aiex.cert.job(0) {
        aiex.cert.write32(0x2000, 1)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.uc_dma_write_des_sync(@shared)
        }
      }
    }
    // Blocks for the same controller share one copy.
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.uc_dma_write_des_sync(@shared)
          aiex.cert.uc_dma_write_des_sync(@other)
        }
      }
    }
    aiex.cert.attach_to_group(2) {
      aiex.cert.page {
        aiex.cert.job(3) {
          aiex.cert.uc_dma_write_des_sync(@shared)
        }
      }
    }
  }
}

// -----

// Device-level jobs and attach_to_group(0) share controller 0's copy.
// Definitions may follow their users.
// CHECK-LABEL: .attach_to_group 0
// CHECK:       START_JOB 4
// CHECK:       START_JOB 5
// CHECK:       EOF
// CHECK:       zero_shared:
// CHECK-NOT:   zero_shared:
// CHECK:       zero_data:
// CHECK-NOT:   zero_data:
// CHECK-LABEL: .attach_to_group 3
// CHECK:       START_JOB 6
// CHECK:       EOF
// CHECK:       zero_shared:
// CHECK:       zero_data:

module {
  aie.device(xcve3858) {
    aiex.cert.page {
      aiex.cert.job(4) {
        aiex.cert.uc_dma_write_des_sync(@zero_shared)
      }
    }
    aiex.cert.attach_to_group(0) {
      aiex.cert.page {
        aiex.cert.job(5) {
          aiex.cert.uc_dma_write_des_sync(@zero_shared)
        }
      }
    }
    aiex.cert.attach_to_group(3) {
      aiex.cert.page {
        aiex.cert.job(6) {
          aiex.cert.uc_dma_write_des_sync(@zero_shared)
        }
      }
    }
    memref.global "private" constant @zero_data : memref<1xi32> = dense<[3]>
    aiex.cert.uc_dma_chain @zero_shared {
      aiex.cert.uc_dma_bd @zero_data, 0x1008, 1, false
    }
  }
}

// -----

// A patched payload lands in the patching group's block, even when the chain
// that pushes it is used only by another group.
// CHECK-LABEL: .attach_to_group 0
// CHECK:         APPLY_OFFSET_57        @patched, 1, 0
// CHECK:       EOF
// CHECK:       patched:
// CHECK-NOT:   patched:
// CHECK-LABEL: .attach_to_group 1
// CHECK:         APPLY_OFFSET_57        @patched, 1, 0
// CHECK-NEXT:    uC_DMA_WRITE_DES_SYNC  @patched_chain
// CHECK:       EOF
// CHECK:       patched_chain:
// CHECK:       patched:
// CHECK-NOT:   patched:

module {
  aie.device(xcve3858) {
    memref.global "private" constant @patched : memref<8xi32> = dense<0>
    aiex.cert.uc_dma_chain @patched_chain {
      aiex.cert.uc_dma_bd @patched, 0x100c, 8, false
    }
    aiex.cert.page {
      aiex.cert.job(7) {
        aiex.cert.apply_offset_57(@patched, 1, 0)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(8) {
          aiex.cert.apply_offset_57(@patched, 1, 0)
          aiex.cert.uc_dma_write_des_sync(@patched_chain)
        }
      }
    }
  }
}

// -----

// A chain used by another controller is not merged into an adjacent push.
// MERGE-LABEL: START_JOB 9
// MERGE:         uC_DMA_WRITE_DES_SYNC  @merge_shared
// MERGE-NEXT:    uC_DMA_WRITE_DES_SYNC  @merge_other
// MERGE:       merge_shared:
// MERGE:       merge_other:
// MERGE-LABEL: .attach_to_group 1
// MERGE:       START_JOB 10
// MERGE:         uC_DMA_WRITE_DES_SYNC  @merge_shared
// MERGE:       EOF
// MERGE:       merge_shared:
// MERGE-NEXT:    UC_DMA_BD       0, 0x00001010, @merge_data, 1, 0, 0
// MERGE:       merge_data:
// MERGE-NOT:   merge_other:

module {
  aie.device(xcve3858) {
    memref.global "private" constant @merge_data : memref<1xi32> = dense<[4]>
    memref.global "private" constant @merge_data2 : memref<1xi32> = dense<[5]>
    aiex.cert.uc_dma_chain @merge_shared {
      aiex.cert.uc_dma_bd @merge_data, 0x1010, 1, false
    }
    aiex.cert.uc_dma_chain @merge_other {
      aiex.cert.uc_dma_bd @merge_data2, 0x1014, 1, false
    }
    aiex.cert.page {
      aiex.cert.job(9) {
        aiex.cert.uc_dma_write_des_sync(@merge_shared)
        aiex.cert.uc_dma_write_des_sync(@merge_other)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(10) {
          aiex.cert.uc_dma_write_des_sync(@merge_shared)
        }
      }
    }
  }
}
