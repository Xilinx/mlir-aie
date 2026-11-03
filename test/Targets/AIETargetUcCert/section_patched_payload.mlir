// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A main-stream patch needs its payload even if only a section uses the chain.
// RUN: rm -rf section_patched_payload && mkdir -p section_patched_payload
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=section_patched_payload %s -o section_patched_payload/main.asm
// RUN: FileCheck %s --check-prefix=MAIN < section_patched_payload/main.asm
// RUN: FileCheck %s --check-prefix=SECTION < section_patched_payload/patched.asm

// MAIN: APPLY_OFFSET_57        @data, 1, 2
// MAIN: EOF
// MAIN-NOT: owned:
// MAIN: data:
// MAIN-NEXT: .long           0x00000001
// MAIN-NOT: data:
// SECTION: APPLY_OFFSET_57        @data, 1, 2
// SECTION: uC_DMA_WRITE_DES_SYNC  @owned
// SECTION: EOF
// SECTION: owned:
// SECTION: UC_DMA_BD {{.*}}@data
// SECTION: data:
// SECTION-NEXT: .long           0x00000001
// SECTION-NOT: data:

module {
  aie.device(xcve3858) {
    memref.global "private" constant @data : memref<9xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9]>
    aiex.cert.uc_dma_chain @owned {
      aiex.cert.uc_dma_bd @data, 0x1000, 9, false
    }
    aiex.cert.section @patched {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.apply_offset_57(@data, 1, 2)
          aiex.cert.uc_dma_write_des_sync(@owned)
        }
      }
    }
    aiex.cert.page {
      aiex.cert.job(2) {
        aiex.cert.apply_offset_57(@data, 1, 2)
      }
    }
  }
}
