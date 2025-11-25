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
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %lock_1_1 = aie.lock(%tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_0 = aie.lock(%tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_1 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_2 = aie.lock(%tile_2_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf8 = aie.buffer(%tile_0_1) {sym_name = "buf8"} : memref<12xi32, 1 : i32> 
    %buf7 = aie.buffer(%tile_0_1) {sym_name = "buf7"} : memref<5xi32, 1 : i32> 
    %buf6 = aie.buffer(%tile_1_1) {sym_name = "buf6"} : memref<12xi32, 1 : i32> 
    %buf5 = aie.buffer(%tile_2_1) {sym_name = "buf5"} : memref<9xi32, 1 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<12xi32, 2 : i32> 
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<5xi32, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<9xi32, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<12xi32, 2 : i32> 
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<5xi32, 2 : i32>, 0, 5)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb4
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb7, repeat_count = 7)
    ^bb3:  // pred: ^bb2
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<12xi32, 2 : i32>, 0, 12)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // 3 preds: ^bb1, ^bb3, ^bb5
      aie.end
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb4, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<12xi32, 2 : i32>, 0, 12)
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb2
      %3 = aie.dma_start(MM2S, 0, ^bb8, ^bb5, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<9xi32, 2 : i32>, 0, 9)
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c12 = arith.constant 12 : index
      %c9 = arith.constant 9 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c9 step %c1 {
        memref.store %c0_i32, %buf2[%arg0] : memref<9xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c5 step %c1 {
        %0 = memref.load %buf2[%c0] : memref<9xi32, 2 : i32>
        %1 = memref.load %buf3[%arg0] : memref<5xi32, 2 : i32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %buf2[%c0] : memref<9xi32, 2 : i32>
      }
      aie.use_lock(%lock_0_2_4, Release, 1)
      scf.for %arg0 = %c1 to %c9 step %c1 {
        aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c12 step %c1 {
          %0 = memref.load %buf4[%arg1] : memref<12xi32, 2 : i32>
          %1 = memref.load %buf1[%arg1] : memref<12xi32, 2 : i32>
          %2 = memref.load %buf2[%arg0] : memref<9xi32, 2 : i32>
          %3 = arith.addi %0, %1 : i32
          %4 = arith.addi %3, %2 : i32
          memref.store %4, %buf2[%arg0] : memref<9xi32, 2 : i32>
        }
        aie.use_lock(%lock_0_2, Release, 1)
        aie.use_lock(%lock_0_2_4, Release, 1)
      }
      aie.use_lock(%lock_0_2_7, Release, 1)
      cf.br ^bb1
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_2_0, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_2_1, DMA : 0)
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<9xi32, 1 : i32>, 0, 9)
      aie.use_lock(%lock_2_1_2, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<9xi32, 1 : i32>, 0, 9)
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<5xi32, 1 : i32>, 0, 5)
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb5, repeat_count = 7)
    ^bb3:  // pred: ^bb2
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<12xi32, 1 : i32>, 0, 12)
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // 5 preds: ^bb1, ^bb3, ^bb6, ^bb7, ^bb8
      aie.end
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<5xi32, 1 : i32>, 0, 5)
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 7)
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<12xi32, 1 : i32>, 0, 12)
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<12xi32, 1 : i32>, 0, 12)
      aie.use_lock(%lock_1_1_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<12xi32, 1 : i32>, 0, 12)
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    }
    aie.shim_dma_allocation @airMemcpyId12 (%tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId4 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId5 (%tile_1_0, MM2S, 0)
    aie.runtime_sequence @six(%arg0: memref<5xi32>, %arg1: memref<96xi32>, %arg2: memref<96xi32>, %arg3: memref<9xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 5][0, 0, 0, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<5xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 96][0, 0, 0, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<96xi32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 96][0, 0, 0, 1]) {id = 0 : i64, metadata = @airMemcpyId5} : memref<96xi32>
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 0][1, 1, 1, 9][0, 0, 0, 1]) {id = 0 : i64, metadata = @airMemcpyId12} : memref<9xi32>
      aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}

