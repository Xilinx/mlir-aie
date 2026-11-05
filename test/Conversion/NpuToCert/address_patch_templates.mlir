// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// cert.apply_offset_57 adds a buffer address to the BD template in memory,
// where an address_patch writes buffer address + arg_plus into the register.
// So each patched template is private to its BD and holds arg_plus.

// RUN: aie-opt --aie-dma-tasks-to-npu --aie-dma-to-npu --aie-npu-to-cert --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-dma-tasks-to-npu --aie-dma-to-npu --aie-npu-to-cert --split-input-file %s | aie-translate --aie-cert-to-asm --split-input-file | FileCheck %s --check-prefix=ASM

// Two copies with the same BD shape share one template after dma-to-npu. Each
// BD gets its own, and the second carries its 32-byte offset.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @[[A:[a-z_0-9]+]] : memref<8xi32> = dense<[16, 0, 0, 0,
// CHECK-DAG: memref.global "private" constant @[[B:[a-z_0-9]+]] : memref<8xi32> = dense<[16, 32, 0, 0,
// CHECK-DAG: aiex.cert.uc_dma_bd @[[A]], 36864, 8, false
// CHECK-DAG: aiex.cert.uc_dma_bd @[[B]], 36912, 8, false
// CHECK: aiex.cert.apply_offset_57(@[[A]], 1, 0)
// CHECK: aiex.cert.apply_offset_57(@[[B]], 1, 1)
// ASM: APPLY_OFFSET_57        @[[ASMA:[a-z_0-9]+]], 1, 0
// ASM: APPLY_OFFSET_57        @[[ASMB:[a-z_0-9]+]], 1, 1
// ASM: [[ASMA]]:
// ASM-NEXT: .long           0x00000010
// ASM-NEXT: .long           0x00000000
// ASM: [[ASMB]]:
// ASM-NEXT: .long           0x00000010
// ASM-NEXT: .long           0x00000020
module {
  aie.device(xcve3858) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%shim, MM2S, 0)
    aie.shim_dma_allocation @in1(%shim, MM2S, 1)
    aie.runtime_sequence @seq(%a: memref<64xi32>, %b: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 8][1, 1, 1, 16][0, 0, 0, 1]) {id = 1 : i64, metadata = @in1} : memref<64xi32>
    }
  }
}

// -----

// A DMA task already writes its offset into the template as well as into
// arg_plus. Setting the word, rather than adding to it, keeps it 32.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: memref.global "private" constant @[[T:[a-z_0-9]+]] : memref<8xi32> = dense<[16, 32, 0, 0,
// CHECK: aiex.cert.apply_offset_57(@[[T]], 1, 0)
// ASM: APPLY_OFFSET_57        @[[ASMT:[a-z_0-9]+]], 1, 0
// ASM: [[ASMT]]:
// ASM-NEXT: .long           0x00000010
// ASM-NEXT: .long           0x00000020
module {
  aie.device(xcve3858) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @seq(%a: memref<64xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%a : memref<64xi32> offset = 8 len = 16) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
