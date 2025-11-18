//===- bd_chain_on_memtile/join_with_iterations.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:     %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 3)
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       aie.use_lock(%input_small_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_0 : memref<1024xui8>, 0, 256)
// CHECK:       aie.use_lock(%input_small_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%input_small_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_1 : memref<1024xui8>, 0, 256)
// CHECK:       aie.use_lock(%input_small_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb8, repeat_count = 5)
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%input_medium_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_0 : memref<1024xui8>, 256, 384)
// CHECK:       aie.use_lock(%input_medium_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%input_medium_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_1 : memref<1024xui8>, 256, 384)
// CHECK:       aie.use_lock(%input_medium_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.end
// CHECK:     ^bb8:  // pred: ^bb4
// CHECK:       %2 = aie.dma_start(S2MM, 2, ^bb9, ^bb12, repeat_count = 7)
// CHECK:     ^bb9:  // pred: ^bb8
// CHECK:       aie.use_lock(%input_large_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_0 : memref<1024xui8>, 640, 384)
// CHECK:       aie.use_lock(%input_large_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(%input_large_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_1 : memref<1024xui8>, 640, 384)
// CHECK:       aie.use_lock(%input_large_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       aie.end
// CHECK:     ^bb12:  // pred: ^bb8
// CHECK:       %3 = aie.dma_start(MM2S, 0, ^bb13, ^bb15)
// CHECK:     ^bb13:  // 2 preds: ^bb12, ^bb14
// CHECK:       aie.use_lock(%combined_output_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_0 : memref<1024xui8>, 0, 1024)
// CHECK:       aie.use_lock(%combined_output_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb14
// CHECK:     ^bb14:  // pred: ^bb13
// CHECK:       aie.use_lock(%combined_output_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%combined_output_buff_1 : memref<1024xui8>, 0, 1024)
// CHECK:       aie.use_lock(%combined_output_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb13
// CHECK:     ^bb15:  // pred: ^bb12
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%input_small_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%input_small_buff_0 : memref<256xui8>, 0, 256)
// CHECK:       aie.use_lock(%input_small_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%input_small_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%input_small_buff_1 : memref<256xui8>, 0, 256)
// CHECK:       aie.use_lock(%input_small_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_3 = aie.mem(%tile_0_3) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%input_medium_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%input_medium_buff_0 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%input_medium_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%input_medium_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%input_medium_buff_1 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%input_medium_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_4 = aie.mem(%tile_0_4) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%input_large_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%input_large_buff_0 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%input_large_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%input_large_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%input_large_buff_1 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%input_large_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    
    aie.objectfifo @input_small(%tile_0_2, {%mem_tile_0_1}, 2 : i32) {iter_count = 4 : i32} : !aie.objectfifo<memref<256xui8>> 
    aie.objectfifo @input_medium(%tile_0_3, {%mem_tile_0_1}, 2 : i32) {iter_count = 6 : i32} : !aie.objectfifo<memref<384xui8>> 
    aie.objectfifo @input_large(%tile_0_4, {%mem_tile_0_1}, 2 : i32) {iter_count = 8 : i32} : !aie.objectfifo<memref<384xui8>> 
    
    aie.objectfifo @combined_output(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>> 
    
    aie.objectfifo.link [@input_small, @input_medium, @input_large] -> [@combined_output]([0, 256, 640] [])
  }
}