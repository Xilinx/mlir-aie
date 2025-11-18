//===- bd_chain_on_memtile/split_with_iterations.mlir ----------------------------*- MLIR -*-===//
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
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_0 : memref<1024xui8>, 0, 1024)
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_1 : memref<1024xui8>, 0, 1024)
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6, repeat_count = 2)
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_0 : memref<1024xui8>, 0, 256)
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_1 : memref<1024xui8>, 0, 256)
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9, repeat_count = 3)
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_0 : memref<1024xui8>, 256, 384)
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_1 : memref<1024xui8>, 256, 384)
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb9:  // pred: ^bb6
// CHECK:       %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12, repeat_count = 4)
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_0 : memref<1024xui8>, 640, 384)
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       aie.use_lock(%large_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_input_cons_buff_1 : memref<1024xui8>, 640, 384)
// CHECK:       aie.use_lock(%large_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb12:  // pred: ^bb9
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%small_output_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%small_output_cons_buff_0 : memref<256xui8>, 0, 256)
// CHECK:       aie.use_lock(%small_output_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%small_output_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%small_output_cons_buff_1 : memref<256xui8>, 0, 256)
// CHECK:       aie.use_lock(%small_output_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_3 = aie.mem(%tile_0_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%medium_output_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%medium_output_cons_buff_0 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%medium_output_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%medium_output_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%medium_output_cons_buff_1 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%medium_output_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_4 = aie.mem(%tile_0_4) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%large_output_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_output_cons_buff_0 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%large_output_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%large_output_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%large_output_cons_buff_1 : memref<384xui8>, 0, 384)
// CHECK:       aie.use_lock(%large_output_cons_cons_lock_0, Release, 1)
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
    
    aie.objectfifo @large_input(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xui8>> 
    
    aie.objectfifo @small_output(%mem_tile_0_1, {%tile_0_2}, 2 : i32) {iter_count = 3 : i32} : !aie.objectfifo<memref<256xui8>> 
    aie.objectfifo @medium_output(%mem_tile_0_1, {%tile_0_3}, 2 : i32) {iter_count = 4 : i32} : !aie.objectfifo<memref<384xui8>> 
    aie.objectfifo @large_output(%mem_tile_0_1, {%tile_0_4}, 2 : i32) {iter_count = 5 : i32} : !aie.objectfifo<memref<384xui8>> 
    
    aie.objectfifo.link [@large_input] -> [@small_output, @medium_output, @large_output]([] [0, 256, 640])
  }
}