//===- bd_chain_on_memtile/repeat_chain.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %[[SHIM_TILE:.*]] = aie.tile(0, 0)
// CHECK:     %[[MEM_TILE:.*]] = aie.tile(0, 1)
// CHECK:     %[[IN_CONS_BUFF_0:.*]] = aie.buffer(%[[MEM_TILE]]) {sym_name = "in_cons_buff_0"} : memref<1024xi32>
// CHECK:     %[[IN_CONS_BUFF_1:.*]] = aie.buffer(%[[MEM_TILE]]) {sym_name = "in_cons_buff_1"} : memref<1024xi32>
// CHECK:     %[[IN_CONS_PROD_LOCK:.*]] = aie.lock(%[[MEM_TILE]], 0) {init = 2 : i32, sym_name = "in_cons_prod_lock_0"}
// CHECK:     %[[IN_CONS_CONS_LOCK:.*]] = aie.lock(%[[MEM_TILE]], 1) {init = 0 : i32, sym_name = "in_cons_cons_lock_0"}
// CHECK:     aie.flow(%[[SHIM_TILE]], DMA : 0, %[[MEM_TILE]], DMA : 0)
// CHECK:     aie.flow(%[[MEM_TILE]], DMA : 0, %[[SHIM_TILE]], DMA : 0)
// CHECK:     %[[MEMTILE_DMA:.*]] = aie.memtile_dma(%[[MEM_TILE]]) {
// CHECK:       %[[S2MM_START:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4, repeat_count = 4)
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       aie.use_lock(%[[IN_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN_CONS_BUFF_0]] : memref<1024xi32>, 0, 1024)
// CHECK:       aie.use_lock(%[[IN_CONS_CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[IN_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN_CONS_BUFF_1]] : memref<1024xi32>, 0, 1024)
// CHECK:       aie.use_lock(%[[IN_CONS_CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       %[[MM2S_START:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:     ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK:       aie.use_lock(%[[IN_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN_CONS_BUFF_0]] : memref<1024xi32>, 0, 1024)
// CHECK:       aie.use_lock(%[[IN_CONS_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%[[IN_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[IN_CONS_BUFF_1]] : memref<1024xi32>, 0, 1024)
// CHECK:       aie.use_lock(%[[IN_CONS_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb7:  // pred: ^bb4
// CHECK:       aie.end
// CHECK:     }

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32)  {bd_chain_repeat_count = 4 : i32} : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo @in_fwd(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>> 
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])
  }
}