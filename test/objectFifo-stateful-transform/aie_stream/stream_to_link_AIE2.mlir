//===- stream_to_link_AIE2.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @stream_to_link_AIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %shim_pl_tile_1_0 = aie.tile(1, 0)
// CHECK:     %mem_tile_1_1 = aie.tile(1, 1)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %of_out_cons_prod_lock_0 = aie.lock(%shim_pl_tile_1_0, 0) {init = 0 : i32, sym_name = "of_out_cons_prod_lock_0"}
// CHECK:     %of_out_cons_cons_lock_0 = aie.lock(%shim_pl_tile_1_0, 1) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
// CHECK:     %of_stream_cons_buff_0 = aie.buffer(%mem_tile_1_1) {sym_name = "of_stream_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of_stream_cons_buff_1 = aie.buffer(%mem_tile_1_1) {sym_name = "of_stream_cons_buff_1"} : memref<16xi32> 
// CHECK:     %of_stream_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "of_stream_cons_prod_lock_0"}
// CHECK:     %of_stream_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "of_stream_cons_cons_lock_0"}
// CHECK:     aie.flow(%tile_3_3, Core : 0, %mem_tile_1_1, DMA : 0)
// CHECK:     aie.flow(%mem_tile_1_1, DMA : 0, %shim_pl_tile_1_0, DMA : 0)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%of_stream_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_stream_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_stream_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of_stream_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_stream_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_stream_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       aie.use_lock(%of_stream_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_stream_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_stream_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%of_stream_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_stream_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_stream_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of_out_shim_alloc(S2MM, 0, 1)
// CHECK:   }
// CHECK: }

module @stream_to_link_AIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1) 
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile33, {%tile11}, 2 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of_stream] -> [@of_out] ([] [])
  }
}
