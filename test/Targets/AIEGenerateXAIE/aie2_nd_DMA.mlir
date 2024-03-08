//===- aie2_nd_DMA.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
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
      aie.use_lock(%l01_0, "AcquireGreaterEqual", 1)
      aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 128, [<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>])
      aie.use_lock(%l01_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd1:
      aie.use_lock(%l01_1, "AcquireGreaterEqual", 1)
      aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 16)
      aie.use_lock(%l01_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd2:
      aie.use_lock(%l01_2, "AcquireGreaterEqual", 1)
      aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
      aie.use_lock(%l01_3, "Release", 1)
      aie.next_bd ^bd2
    ^bd3:
      aie.use_lock(%l01_3, "AcquireGreaterEqual", 1)
      aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
      aie.use_lock(%l01_2, "Release", 1)
      aie.next_bd ^bd3
    ^end:
      aie.end
  }
 }
}
