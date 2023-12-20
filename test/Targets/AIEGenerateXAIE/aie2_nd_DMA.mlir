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
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[3].AieMlDimDesc = { /* StepSize */ 1, /* Wrap */ 2};
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[2].AieMlDimDesc = { /* StepSize */ 2, /* Wrap */ 3};
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[1].AieMlDimDesc = { /* StepSize */ 4, /* Wrap */ 2};
// CHECK: dma_tile_2_1_bd_0_tensor.Dim[0].AieMlDimDesc = { /* StepSize */ 1, /* Wrap */ 1};
// CHECK: __mlir_aie_try(XAie_DmaSetMultiDimAddr(&(dma_tile21_bd0), &dma_tile_2_1_bd_0_tensor, 0x82000,  /* len */ 128 * 4));

module @aie_module  {
 AIE.device(xcve2302) {
  %t01 = AIE.tile(2, 1)
  %buf01_0 = AIE.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<16xi32>
  %buf01_1 = AIE.buffer(%t01) { address = 1824 : i32, sym_name = "out" } : memref<16xi32>

  %trhesholdValue = arith.constant 100      : i16

  %l01_0 = AIE.lock(%t01, 0) { init = 1 : i32 }
  %l01_1 = AIE.lock(%t01, 1)
  %l01_2 = AIE.lock(%t01, 2) { init = 1 : i32 }
  %l01_3 = AIE.lock(%t01, 3)

  %m01 = AIE.memtile_dma(%t01) {
      %srcDma = AIE.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %memSrcDma = AIE.dma_start(MM2S, 1, ^bd1, ^dma1)
    ^dma1:
      %memDstDma = AIE.dma_start(S2MM, 1, ^bd2, ^dma2)
    ^dma2:
      %dstDma = AIE.dma_start(MM2S, 0, ^bd3, ^end)
    ^bd0:
      AIE.use_lock(%l01_0, "AcquireGreaterEqual", 1)
      AIE.dma_bd(%buf01_0 : memref<16xi32>, 0, 128, [<wrap = 2, step = 1>, <wrap = 3, step = 2>, <wrap = 2, step = 4>, <wrap = 1, step = 1>])
      AIE.use_lock(%l01_1, "Release", 1)
      AIE.next_bd ^bd0
    ^bd1:
      AIE.use_lock(%l01_1, "AcquireGreaterEqual", 1)
      AIE.dma_bd(%buf01_0 : memref<16xi32>, 0, 16)
      AIE.use_lock(%l01_0, "Release", 1)
      AIE.next_bd ^bd1
    ^bd2:
      AIE.use_lock(%l01_2, "AcquireGreaterEqual", 1)
      AIE.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
      AIE.use_lock(%l01_3, "Release", 1)
      AIE.next_bd ^bd2
    ^bd3:
      AIE.use_lock(%l01_3, "AcquireGreaterEqual", 1)
      AIE.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
      AIE.use_lock(%l01_2, "Release", 1)
      AIE.next_bd ^bd3
    ^end:
      AIE.end
  }
 }
}
