//===- link_test_join_offsets.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @link_distribute_offsets {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK-DAG:     %{{.*}}tile_2_1 = aie.tile(2, 1)
// CHECK-DAG:     %{{.*}}tile_2_2 = aie.tile(2, 2)
// CHECK-DAG:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK-DAG:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG: aie.buffer @[[LINK4_BUFF_0:link4_buff_0]](%{{.*}}tile_2_1) : memref<48xi32>
// CHECK-DAG: aie.buffer @[[LINK4_BUFF_1:link4_buff_1]](%{{.*}}tile_2_1) : memref<48xi32>
// CHECK-DAG: aie.lock @[[LINK4_PROD_LOCK_0:link4_prod_lock_0]](%{{.*}}tile_2_1) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[LINK4_CONS_LOCK_0:link4_cons_lock_0]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG: aie.lock @[[LINK4_PROD_LOCK_1:link4_prod_lock_1]](%{{.*}}tile_2_1) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[LINK4_CONS_LOCK_1:link4_cons_lock_1]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG: aie.lock @[[LINK4_PROD_LOCK_2:link4_prod_lock_2]](%{{.*}}tile_2_1) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[LINK4_CONS_LOCK_2:link4_cons_lock_2]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[LINK3_BUFF_0:link3_buff_0]](%{{.*}}tile_3_3) : memref<12xi32>
// CHECK-DAG: aie.buffer @[[LINK3_BUFF_1:link3_buff_1]](%{{.*}}tile_3_3) : memref<12xi32>
// CHECK-DAG: aie.lock @[[LINK3_PROD_LOCK:link3_prod_lock_0]](%{{.*}}tile_3_3) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[LINK3_CONS_LOCK:link3_cons_lock_0]](%{{.*}}tile_3_3) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[LINK2_BUFF_0:link2_buff_0]](%{{.*}}tile_2_3) : memref<20xi32>
// CHECK-DAG: aie.buffer @[[LINK2_BUFF_1:link2_buff_1]](%{{.*}}tile_2_3) : memref<20xi32>
// CHECK-DAG: aie.lock @[[LINK2_PROD_LOCK:link2_prod_lock_0]](%{{.*}}tile_2_3) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[LINK2_CONS_LOCK:link2_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[LINK1_BUFF_0:link1_buff_0]](%{{.*}}tile_2_2) : memref<4x4xi32>
// CHECK-DAG: aie.buffer @[[LINK1_BUFF_1:link1_buff_1]](%{{.*}}tile_2_2) : memref<4x4xi32>
// CHECK-DAG: aie.lock @[[LINK1_PROD_LOCK:link1_prod_lock_0]](%{{.*}}tile_2_2) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[LINK1_CONS_LOCK:link1_cons_lock_0]](%{{.*}}tile_2_2) {init = 0 : i32}
// CHECK-DAG: aie.flow(%{{.*}}tile_2_2, DMA : 0, %{{.*}}tile_2_1, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_2_3, DMA : 0, %{{.*}}tile_2_1, DMA : 1)
// CHECK-DAG: aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}tile_2_1, DMA : 2)
// CHECK-DAG: aie.flow(%{{.*}}tile_2_1, DMA : 0, %{{.*}}tile_2_0, DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @link4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
// CHECK:     %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[LINK1_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK1_BUFF_0_ACCESS_0:.*]] = aie.get_buffer @[[LINK1_BUFF_0]] : memref<4x4xi32>
// CHECK:       aie.dma_bd(%[[LINK1_BUFF_0_ACCESS_0]] : memref<4x4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK1_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[LINK1_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK1_BUFF_1_ACCESS_1:.*]] = aie.get_buffer @[[LINK1_BUFF_1]] : memref<4x4xi32>
// CHECK:       aie.dma_bd(%[[LINK1_BUFF_1_ACCESS_1]] : memref<4x4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK1_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%{{.*}}tile_2_1) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_0_ACCESS_2:.*]] = aie.get_buffer @[[LINK4_BUFF_0]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0_ACCESS_2]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_1_ACCESS_3:.*]] = aie.get_buffer @[[LINK4_BUFF_1]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1_ACCESS_3]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_0_ACCESS_4:.*]] = aie.get_buffer @[[LINK4_BUFF_0]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0_ACCESS_4]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_1_ACCESS_5:.*]] = aie.get_buffer @[[LINK4_BUFF_1]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1_ACCESS_5]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
// CHECK:     ^bb7:
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_0_ACCESS_6:.*]] = aie.get_buffer @[[LINK4_BUFF_0]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0_ACCESS_6]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_2]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_1_ACCESS_7:.*]] = aie.get_buffer @[[LINK4_BUFF_1]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1_ACCESS_7]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_2]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb9:
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(MM2S, 0, ^bb10, ^bb16)
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_0_ACCESS_8:.*]] = aie.get_buffer @[[LINK4_BUFF_0]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0_ACCESS_8]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_0_ACCESS_9:.*]] = aie.get_buffer @[[LINK4_BUFF_0]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0_ACCESS_9]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb12
// CHECK:     ^bb12:
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_0_ACCESS_10:.*]] = aie.get_buffer @[[LINK4_BUFF_0]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0_ACCESS_10]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_2]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb13
// CHECK:     ^bb13:
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_1_ACCESS_11:.*]] = aie.get_buffer @[[LINK4_BUFF_1]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1_ACCESS_11]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb14
// CHECK:     ^bb14:
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_1_ACCESS_12:.*]] = aie.get_buffer @[[LINK4_BUFF_1]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1_ACCESS_12]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb15
// CHECK:     ^bb15:
// CHECK:       aie.use_lock(@[[LINK4_CONS_LOCK_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK4_BUFF_1_ACCESS_13:.*]] = aie.get_buffer @[[LINK4_BUFF_1]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1_ACCESS_13]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK4_PROD_LOCK_2]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb16:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %[[CHECK_SSA_5:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[LINK2_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK2_BUFF_0_ACCESS_14:.*]] = aie.get_buffer @[[LINK2_BUFF_0]] : memref<20xi32>
// CHECK:       aie.dma_bd(%[[LINK2_BUFF_0_ACCESS_14]] : memref<20xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK2_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[LINK2_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK2_BUFF_1_ACCESS_15:.*]] = aie.get_buffer @[[LINK2_BUFF_1]] : memref<20xi32>
// CHECK:       aie.dma_bd(%[[LINK2_BUFF_1_ACCESS_15]] : memref<20xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK2_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %[[CHECK_SSA_6:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[LINK3_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK3_BUFF_0_ACCESS_16:.*]] = aie.get_buffer @[[LINK3_BUFF_0]] : memref<12xi32>
// CHECK:       aie.dma_bd(%[[LINK3_BUFF_0_ACCESS_16]] : memref<12xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK3_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[LINK3_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[LINK3_BUFF_1_ACCESS_17:.*]] = aie.get_buffer @[[LINK3_BUFF_1]] : memref<12xi32>
// CHECK:       aie.dma_bd(%[[LINK3_BUFF_1_ACCESS_17]] : memref<12xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[LINK3_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @link_distribute_offsets {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
        aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

        aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
    }
}
