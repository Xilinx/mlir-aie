// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An address_patch matches the BD blockwrite before it by the blockwrite's
// absolute address, so a BD written relative to its tile is still patched.

// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite --implicit-check-not=aiex.npu.address_patch
// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | aie-translate --aie-cert-to-asm --split-input-file | FileCheck %s --check-prefix=ASM

// Shim tile (1, 0) BD 0 on xcve3858 is at 0x02009000; its address word is
// the second one.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: memref.global "private" constant @bd : memref<8xi32> = dense<[16, 32, 0, 0, -1073741824, 33554432, 0, 33554432]>
// CHECK: aiex.cert.uc_dma_bd @bd, 33591296, 8, false
// CHECK: aiex.cert.apply_offset_57(@bd, 1, 0)
// ASM: APPLY_OFFSET_57        @bd, 1, 0
// ASM: UC_DMA_BD 0, 0x02009000, @bd, 8, 0, 0
// ASM: bd:
// ASM-NEXT: .long           0x00000010
// ASM-NEXT: .long           0x00000020
aie.device(xcve3858) {
  memref.global "private" constant @bd : memref<8xi32> = dense<[16, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
  aie.runtime_sequence @seq(%a: memref<64xi32>) {
    %0 = memref.get_global @bd : memref<8xi32>
    aiex.npu.blockwrite(%0) {address = 0x9000 : ui32, column = 1 : i32, row = 0 : i32} : memref<8xi32>
    %c32 = arith.constant 32 : i32
    aiex.npu.address_patch(%c32 : i32) {addr = 0x02009004 : ui32, arg_idx = 0 : i32}
  }
}

// -----

// The same on npu2, where shim tile (1, 0) BD 0 is at 0x0201D000.
// CHECK-LABEL: aie.device(npu2)
// CHECK: memref.global "private" constant @bd : memref<8xi32> = dense<[16, 32, 0, 0, -1073741824, 33554432, 0, 33554432]>
// CHECK: aiex.cert.uc_dma_bd @bd, 33673216, 8, false
// CHECK: aiex.cert.apply_offset_57(@bd, 1, 0)
// ASM: APPLY_OFFSET_57        @bd, 1, 0
// ASM: UC_DMA_BD 0, 0x0201d000, @bd, 8, 0, 0
// ASM: bd:
// ASM-NEXT: .long           0x00000010
// ASM-NEXT: .long           0x00000020
aie.device(npu2) {
  memref.global "private" constant @bd : memref<8xi32> = dense<[16, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
  aie.runtime_sequence @seq(%a: memref<64xi32>) {
    %0 = memref.get_global @bd : memref<8xi32>
    aiex.npu.blockwrite(%0) {address = 0x1D000 : ui32, column = 1 : i32, row = 0 : i32} : memref<8xi32>
    %c32 = arith.constant 32 : i32
    aiex.npu.address_patch(%c32 : i32) {addr = 0x0201D004 : ui32, arg_idx = 0 : i32}
  }
}
