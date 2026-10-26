// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared chains resolve separately in each section and in the main stream.
// RUN: rm -rf section_chain_ownership && mkdir -p section_chain_ownership
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=section_chain_ownership %s -o section_chain_ownership/main.asm
// RUN: FileCheck %s --check-prefix=MAIN < section_chain_ownership/main.asm
// RUN: FileCheck %s --check-prefix=FIRST < section_chain_ownership/first.asm
// RUN: FileCheck %s --check-prefix=SECOND < section_chain_ownership/second.asm
// RUN: FileCheck %s --check-prefix=EMPTY < section_chain_ownership/empty.asm

// MAIN-NOT: exclusive:
// MAIN-NOT: section_shared:
// MAIN: uC_DMA_WRITE_DES_SYNC  @main_shared
// MAIN: EOF
// MAIN-NOT: exclusive:
// MAIN-NOT: section_shared:
// MAIN: main_shared:
// MAIN: UC_DMA_BD {{.*}}@data
// MAIN: data:
// MAIN: .long           0x00000001
// MAIN-NEXT: .long           0x00000002
// MAIN-NOT: data:

// FIRST: uC_DMA_WRITE_DES_SYNC  @exclusive
// FIRST: uC_DMA_WRITE_DES_SYNC  @section_shared
// FIRST: uC_DMA_WRITE_DES_SYNC  @main_shared
// FIRST: uC_DMA_WRITE_DES_SYNC  @exclusive
// FIRST: EOF
// FIRST: exclusive:
// FIRST: UC_DMA_BD {{.*}}@data
// FIRST: section_shared:
// FIRST: UC_DMA_BD {{.*}}@data
// FIRST: main_shared:
// FIRST: UC_DMA_BD {{.*}}@data
// FIRST: data:
// FIRST-NEXT: .long           0x00000001
// FIRST-NEXT: .long           0x00000002
// FIRST-NOT: data:
// FIRST-NOT: exclusive:

// SECOND: uC_DMA_WRITE_DES_SYNC  @section_shared
// SECOND: EOF
// SECOND: section_shared:
// SECOND: UC_DMA_BD {{.*}}@data
// SECOND: data:
// EMPTY: WRITE_32
// EMPTY: EOF
// EMPTY-NOT: UC_DMA_BD
// EMPTY-NOT: data:

module {
  aie.device(xcve3858) {
    memref.global "private" constant @data : memref<2xi32> = dense<[1, 2]>
    aiex.cert.uc_dma_chain @exclusive {
      aiex.cert.uc_dma_bd @data, 0x1000, 2, false
    }
    aiex.cert.uc_dma_chain @section_shared {
      aiex.cert.uc_dma_bd @data, 0x2000, 2, false
    }
    aiex.cert.uc_dma_chain @main_shared {
      aiex.cert.uc_dma_bd @data, 0x3000, 2, false
    }
    aiex.cert.section @first {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.uc_dma_write_des_sync(@exclusive)
          aiex.cert.uc_dma_write_des_sync(@section_shared)
          aiex.cert.uc_dma_write_des_sync(@main_shared)
          aiex.cert.uc_dma_write_des_sync(@exclusive)
        }
      }
    }
    aiex.cert.section @second {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.uc_dma_write_des_sync(@section_shared)
        }
      }
    }
    aiex.cert.section @empty {
      aiex.cert.page {
        aiex.cert.job(3) {
          aiex.cert.write32(0x4000, 3)
        }
      }
    }
    aiex.cert.page {
      aiex.cert.job(5) {
        aiex.cert.uc_dma_write_des_sync(@main_shared)
      }
    }
  }
}
