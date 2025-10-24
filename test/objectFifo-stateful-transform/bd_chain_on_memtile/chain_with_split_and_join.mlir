//===- bd_chain_on_memtile/ichain_with_split_and_join.mlir ----------------*- MLIR -*-===//
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
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb9, repeat_count = 7)
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%in_cons_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       aie.end
// CHECK:     ^bb9:  // pred: ^bb5
// CHECK:       %2 = aie.dma_start(MM2S, 1, ^bb10, ^bb13, repeat_count = 7)
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_0 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       aie.use_lock(%in_cons_cons_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%in_cons_buff_1 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%in_cons_prod_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb12
// CHECK:     ^bb12:  // pred: ^bb11
// CHECK:       aie.end
// CHECK:     ^bb13:  // pred: ^bb9
// CHECK:       %3 = aie.dma_start(S2MM, 1, ^bb14, ^bb17, repeat_count = 7)
// CHECK:     ^bb14:  // pred: ^bb13
// CHECK:       aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%out_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb15
// CHECK:     ^bb15:  // pred: ^bb14
// CHECK:       aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%out_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb16
// CHECK:     ^bb16:  // pred: ^bb15
// CHECK:       aie.end
// CHECK:     ^bb17:  // pred: ^bb13
// CHECK:       %4 = aie.dma_start(S2MM, 2, ^bb18, ^bb21, repeat_count = 7)
// CHECK:     ^bb18:  // pred: ^bb17
// CHECK:       aie.use_lock(%out_prod_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%out_cons_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb19
// CHECK:     ^bb19:  // pred: ^bb18
// CHECK:       aie.use_lock(%out_prod_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%out_cons_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb20
// CHECK:     ^bb20:  // pred: ^bb19
// CHECK:       aie.end
// CHECK:     ^bb21:  // pred: ^bb17
// CHECK:       %5 = aie.dma_start(MM2S, 2, ^bb22, ^bb26)
// CHECK:     ^bb22:  // 2 preds: ^bb21, ^bb25
// CHECK:       aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%out_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb23
// CHECK:     ^bb23:  // pred: ^bb22
// CHECK:       aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_0 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%out_prod_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb24
// CHECK:     ^bb24:  // pred: ^bb23
// CHECK:       aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8>, 0, 512)
// CHECK:       aie.use_lock(%out_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb25
// CHECK:     ^bb25:  // pred: ^bb24
// CHECK:       aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%out_buff_1 : memref<1024xui8>, 512, 512)
// CHECK:       aie.use_lock(%out_prod_lock_1, Release, 1)
// CHECK:       aie.next_bd ^bb22
// CHECK:     ^bb26:  // pred: ^bb21
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%split_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%split_0_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%split_0_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%split_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%split_0_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%split_0_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       aie.use_lock(%join_0_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%join_0_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%join_0_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%join_0_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%join_0_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%join_0_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_0_3 = aie.mem(%tile_0_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%split_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%split_1_cons_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%split_1_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%split_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%split_1_cons_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%split_1_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       aie.use_lock(%join_1_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%join_1_buff_0 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%join_1_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%join_1_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%join_1_buff_1 : memref<512xui8>, 0, 512)
// CHECK:       aie.use_lock(%join_1_prod_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       aie.end
// CHECK:     }

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @split_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) {bd_chain_iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>> 
    aie.objectfifo @split_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) {bd_chain_iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>> 
    aie.objectfifo.link [@in] -> [@split_0, @split_1]([] [0, 512])
    aie.objectfifo @join_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) {bd_chain_iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>> 
    aie.objectfifo @join_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) {bd_chain_iter_count = 8 : i32} : !aie.objectfifo<memref<512xui8>> 
    aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo.link [@join_0, @join_1] -> [@out]([0, 512] [])
  }
}
