// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A blockwrite addressed relative to a tile or a buffer is lowered at the
// absolute address it targets, including when it is split.

// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite
// RUN: aie-opt --aie-dma-to-npu --aie-npu-to-cert --split-input-file %s | FileCheck %s --implicit-check-not=aiex.npu.blockwrite
// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | aie-translate --aie-cert-to-asm --split-input-file | FileCheck %s --check-prefix=ASM

// Tile coordinates replace any upper address bits; a buffer offset is in
// words. An address without coordinates stays absolute.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.uc_dma_bd @data, 35651840, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 35651840, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 305398016, 4, false
// CHECK: aiex.cert.uc_dma_bd @data, 35652612, 4, false
// ASM: UC_DMA_BD 0, 0x02200100, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x02200100, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x12340100, @data, 4, 0, 0
// ASM: UC_DMA_BD 0, 0x02200404, @data, 4, 0, 0
aie.device(npu2) {
  %tile = aie.tile(1, 2)
  %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "buffer"} : memref<16xi32>
  memref.global "private" constant @data : memref<4xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x12300100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 0x12340100 : ui32} : memref<4xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4xi32>
  }
}

// -----

// 4096 words split recursively into four 1024-word transfers. Every child
// keeps the tile base and advances by the bytes before it.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35651840, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35655936, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35660032, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35664128, 1024, false
// ASM: UC_DMA_BD 0, 0x02200100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02201100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02202100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02203100, @{{[^,]+}}, 1024, 0, 0
aie.device(npu2) {
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 0x100 : ui32, column = 1 : i32, row = 2 : i32} : memref<4096xi32>
  }
}

// -----

// A split buffer-relative write resolves the buffer and word offset once.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35652612, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35656708, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35660804, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 35664900, 1024, false
// ASM: UC_DMA_BD 0, 0x02200404, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02201404, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02202404, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x02203404, @{{[^,]+}}, 1024, 0, 0
aie.device(npu2) {
  %tile = aie.tile(1, 2)
  %buffer = aie.buffer(%tile) {address = 1024 : i32, sym_name = "buffer"} : memref<4097xi32>
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 1 : ui32, buffer = @buffer} : memref<4096xi32>
  }
}

// -----

// Splitting an absolute address keeps all of its upper bits.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305398016, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305402112, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305406208, 1024, false
// CHECK: aiex.cert.uc_dma_bd @{{[^,]+}}, 305410304, 1024, false
// ASM: UC_DMA_BD 0, 0x12340100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x12341100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x12342100, @{{[^,]+}}, 1024, 0, 0
// ASM: UC_DMA_BD 0, 0x12343100, @{{[^,]+}}, 1024, 0, 0
aie.device(npu2) {
  memref.global "private" constant @data : memref<4096xi32> = dense<7>
  aie.runtime_sequence @seq() {
    %data = memref.get_global @data : memref<4096xi32>
    aiex.npu.blockwrite(%data) {address = 0x12340100 : ui32} : memref<4096xi32>
  }
}
