// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Emit the asynchronous uC DMA push and its wait, and place the pushed chain
// wherever AIEBU resolves it: in each consuming controller group's block and
// in each consuming section's file.
// RUN: rm -rf uc_dma_async && mkdir -p uc_dma_async
// RUN: aie-translate --aie-cert-to-asm --cert-section-dir=uc_dma_async %s -o uc_dma_async/main.asm
// RUN: FileCheck %s --check-prefix=MAIN < uc_dma_async/main.asm
// RUN: FileCheck %s --check-prefix=SECTION < uc_dma_async/async_section.asm

// MAIN-LABEL: .attach_to_group 0
// MAIN: START_JOB 1
// MAIN-NEXT: UC_DMA_WRITE_DES       $r2, @group_chain
// MAIN-NEXT: WAIT_UC_DMA            $r2
// MAIN: EOF
// MAIN: group_chain:
// MAIN-NEXT: UC_DMA_BD       0, 0x00001000, @data, 1, 0, 0
// MAIN-NOT: section_chain:
// MAIN: data:
// MAIN-LABEL: .attach_to_group 1
// MAIN: START_JOB 2
// MAIN-NEXT: UC_DMA_WRITE_DES       $r5, @group_chain
// MAIN-NEXT: WAIT_UC_DMA            $r5
// MAIN: EOF
// MAIN: group_chain:
// MAIN: data:

// SECTION: START_JOB 3
// SECTION-NEXT: UC_DMA_WRITE_DES       $r0, @section_chain
// SECTION-NEXT: WAIT_UC_DMA            $r0
// SECTION: EOF
// SECTION: section_chain:
// SECTION-NEXT: UC_DMA_BD       0, 0x00002000, @data, 1, 0, 0
// SECTION: data:

module {
  aie.device(xcve3858) {
    memref.global "private" constant @data : memref<1xi32> = dense<7>
    aiex.cert.uc_dma_chain @group_chain {
      aiex.cert.uc_dma_bd @data, 0x1000, 1, false
    }
    aiex.cert.uc_dma_chain @section_chain {
      aiex.cert.uc_dma_bd @data, 0x2000, 1, false
    }
    aiex.cert.section @async_section {
      aiex.cert.page {
        aiex.cert.job(3) {
          aiex.cert.uc_dma_write_des(0, @section_chain)
          aiex.cert.wait_uc_dma(0)
        }
      }
    }
    aiex.cert.page {
      aiex.cert.job(1) {
        aiex.cert.uc_dma_write_des(2, @group_chain)
        aiex.cert.wait_uc_dma(2)
      }
    }
    aiex.cert.attach_to_group(1) {
      aiex.cert.page {
        aiex.cert.job(2) {
          aiex.cert.uc_dma_write_des(5, @group_chain)
          aiex.cert.wait_uc_dma(5)
        }
      }
    }
  }
}
