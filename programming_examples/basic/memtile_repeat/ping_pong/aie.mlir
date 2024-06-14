//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<1024xi32>
    memref.global "public" @out : memref<1024xi32>
    memref.global "public" @in_cons : memref<1024xi32>
    memref.global "public" @in : memref<1024xi32>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock"}
    %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %in_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "in_cons_buff_0"} : memref<1024xi32> 
    %in_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 4 : i32, sym_name = "in_cons_prod_lock"}
    %in_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock"}
    %in_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    aie.shim_dma_allocation @in(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in_cons_prod_lock, AcquireGreaterEqual, 4)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_cons_cons_lock, Release, 4)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5, repeat_count = 1)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // 2 preds: ^bb3
      aie.use_lock(%in_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      aie.end
    }
  }
}
