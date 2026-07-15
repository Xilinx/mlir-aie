//===- aie2_nd_DMA.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: dma_tile_2_1_bd_0_tensor.NumDim = 4;
// CHECK: dma_tile_2_1_bd_0_tensor.Dim =__mlir_aie_alloc_dim_desc(4);
// CHECK: if(NULL == dma_tile_2_1_bd_0_tensor.Dim){
// CHECK:   return 1;
// CHECK: }
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[3].AieMlDimDesc = { /* Stride */ 1, /* Size */ 2};
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[2].AieMlDimDesc = { /* Stride */ 2, /* Size */ 3};
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[1].AieMlDimDesc = { /* Stride */ 4, /* Size */ 2};
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[0].AieMlDimDesc = { /* Stride */ 1, /* Size */ 1};
// CHECK: __mlir_aie_try(XAie_DmaSetMultiDimAddr(&(dma_tile21_bd0), &dma_tile_2_1_bd_0_tensor, 0x82000,  /* len */ 512));

module @aie_module  {
 aie.device(xcve2302) {
  %t01 = aie.tile(2, 1)
  %buf01_0 = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<16xi32>
  %buf01_1 = aie.buffer(%t01) { address = 1824 : i32, sym_name = "out" } : memref<16xi32>

  %trhesholdValue = arith.constant 100      : i16

  %l01_0 = aie.lock(%t01, 0) { init = 1 : i32 }
  %l01_1 = aie.lock(%t01, 1)
  %l01_2 = aie.lock(%t01, 2) { init = 1 : i32 }
  %l01_3 = aie.lock(%t01, 3)

  %m01 = aie.memtile_dma(%t01) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %memSrcDma = aie.dma_start(MM2S, 1, ^bd1, ^dma1)
    ^dma1:
      %memDstDma = aie.dma_start(S2MM, 1, ^bd2, ^dma2)
    ^dma2:
      %dstDma = aie.dma_start(MM2S, 0, ^bd3, ^end)
    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%l01_0, "AcquireGreaterEqual", %c1_ul1)
      aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 128 sizes = [2, 3, 2, 1] strides = [1, 2, 4, 1])
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%l01_1, "Release", %c1_ul2)
      aie.next_bd ^bd0
    ^bd1:
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%l01_1, "AcquireGreaterEqual", %c1_ul3)
      aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 16)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%l01_0, "Release", %c1_ul4)
      aie.next_bd ^bd1
    ^bd2:
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%l01_2, "AcquireGreaterEqual", %c1_ul5)
      aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16)
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(%l01_3, "Release", %c1_ul6)
      aie.next_bd ^bd2
    ^bd3:
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%l01_3, "AcquireGreaterEqual", %c1_ul7)
      aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16)
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%l01_2, "Release", %c1_ul8)
      aie.next_bd ^bd3
    ^end:
      aie.end
  }
 }
}
