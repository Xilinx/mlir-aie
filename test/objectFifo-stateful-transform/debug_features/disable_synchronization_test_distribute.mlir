//===- disable_synchronization_test_distribute.mlir -------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @disable_sync {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[SHIM_TILE:.*]] = aie.tile(2, 0)
// CHECK:     %[[MEM_TILE:.*]] = aie.tile(2, 1)
// CHECK:     %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:     %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG: %[[LINK3_BUFF:.*]] = aie.buffer(%[[MEM_TILE]]) {sym_name = "link3_buff_0"} : memref<36xi32>
// CHECK-DAG: %[[LINK2_BUFF:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "link2_buff_0"} : memref<20xi32>
// CHECK-DAG: %[[LINK2_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 1 : i32, sym_name = "link2_prod_lock_0"}
// CHECK-DAG: %[[LINK2_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "link2_cons_lock_0"}
// CHECK-DAG: %[[LINK1_BUFF:.*]] = aie.buffer(%[[TILE_2_2]]) {sym_name = "link1_buff_0"} : memref<4x4xi32>
// CHECK-DAG: %[[LINK1_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 0) {init = 1 : i32, sym_name = "link1_prod_lock_0"}
// CHECK-DAG: %[[LINK1_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], 1) {init = 0 : i32, sym_name = "link1_cons_lock_0"}
// CHECK-DAG: aie.flow(%[[TILE_2_2]], DMA : 0, %[[MEM_TILE]], DMA : 0)
// CHECK-DAG: aie.flow(%[[TILE_2_3]], DMA : 0, %[[MEM_TILE]], DMA : 1)
// CHECK-DAG: aie.flow(%[[MEM_TILE]], DMA : 0, %[[SHIM_TILE]], DMA : 0)
// CHECK:     %{{.*}} = aie.mem(%[[TILE_2_2]]) {
// CHECK:       %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK1_BUFF]] : memref<4x4xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %{{.*}} = aie.memtile_dma(%[[MEM_TILE]]) {
// CHECK:       %{{.*}} = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.dma_bd(%[[LINK3_BUFF]] : memref<36xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       %{{.*}} = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:     ^bb3:
// CHECK:       aie.dma_bd(%[[LINK3_BUFF]] : memref<36xi32>, 16, 20)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:
// CHECK:       %{{.*}} = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:     ^bb5:
// CHECK:       aie.dma_bd(%[[LINK3_BUFF]] : memref<36xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:
// CHECK:       aie.dma_bd(%[[LINK3_BUFF]] : memref<36xi32>, 16, 20)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb7:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %{{.*}} = aie.mem(%[[TILE_2_3]]) {
// CHECK:       %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK2_BUFF]] : memref<20xi32>, 0, 20)
// CHECK:       aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @link3_shim_alloc(%{{.*}}, S2MM, 0)
// CHECK:   }
// CHECK: }

module @disable_sync {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 1 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile21, {%tile20}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<36xi32>>

    aie.objectfifo.link [@link1, @link2] -> [@link3] ([0, 16][])
 }
}
