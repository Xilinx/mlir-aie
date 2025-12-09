//===- bd_chain_on_memtile/broadcast_with_iterations.mlir --------------------------------*- MLIR -*-===//
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
// CHECK:       aie.use_lock(%broadcast_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_input_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_input_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%broadcast_input_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_input_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_input_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb7, repeat_count = 7)
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%broadcast_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_input_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%broadcast_input_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_input_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_input_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.end
// CHECK:     ^bb7:  // pred: ^bb3
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%broadcast_output_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_0_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_0_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%broadcast_output_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_0_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_0_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%broadcast_output_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_0_cons_buff_2 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_0_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_3 = aie.mem(%tile_0_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%broadcast_output_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_1_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_1_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%broadcast_output_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_1_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_1_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%broadcast_output_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_1_cons_buff_2 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_1_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_4 = aie.mem(%tile_0_4) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%broadcast_output_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_2_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_2_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%broadcast_output_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_2_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_2_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%broadcast_output_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%broadcast_output_2_cons_buff_2 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%broadcast_output_2_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    
    aie.objectfifo @broadcast_input(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<512xui8>> 
    aie.objectfifo @broadcast_output(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4}, 3 : i32) {iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>> 
    
    aie.objectfifo.link [@broadcast_input] -> [@broadcast_output]([] [])
  }
}
