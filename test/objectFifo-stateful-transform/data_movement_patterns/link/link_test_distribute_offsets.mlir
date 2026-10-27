//===- link_test_distribute_offsets.mlir -----------------------*- MLIR -*-===//
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
// CHECK-DAG:     aie.buffer @[[VAL_0:link4_cons_buff_0]](%{{.*}}tile_3_3) : memref<12xi32>
// CHECK-DAG:     aie.buffer @[[VAL_1:link4_cons_buff_1]](%{{.*}}tile_3_3) : memref<12xi32>
// CHECK-DAG:     aie.lock @[[VAL_2:link4_cons_prod_lock_0]](%{{.*}}tile_3_3) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_3:link4_cons_cons_lock_0]](%{{.*}}tile_3_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_4:link3_cons_buff_0]](%{{.*}}tile_2_3) : memref<20xi32>
// CHECK-DAG:     aie.buffer @[[VAL_5:link3_cons_buff_1]](%{{.*}}tile_2_3) : memref<20xi32>
// CHECK-DAG:     aie.lock @[[VAL_6:link3_cons_prod_lock_0]](%{{.*}}tile_2_3) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_7:link3_cons_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_8:link2_cons_buff_0]](%{{.*}}tile_2_2) : memref<4x4xi32>
// CHECK-DAG:     aie.buffer @[[VAL_9:link2_cons_buff_1]](%{{.*}}tile_2_2) : memref<4x4xi32>
// CHECK-DAG:     aie.lock @[[VAL_10:link2_cons_prod_lock_0]](%{{.*}}tile_2_2) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_11:link2_cons_cons_lock_0]](%{{.*}}tile_2_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_12:link1_cons_buff_0]](%{{.*}}tile_2_1) : memref<48xi32>
// CHECK-DAG:     aie.buffer @[[VAL_13:link1_cons_buff_1]](%{{.*}}tile_2_1) : memref<48xi32>
// CHECK-DAG:     aie.lock @[[VAL_14:link1_cons_prod_lock_0]](%{{.*}}tile_2_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_15:link1_cons_cons_lock_0]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG:     aie.lock @[[VAL_16:link1_cons_prod_lock_1]](%{{.*}}tile_2_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_17:link1_cons_cons_lock_1]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG:     aie.lock @[[VAL_18:link1_cons_prod_lock_2]](%{{.*}}tile_2_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_19:link1_cons_cons_lock_2]](%{{.*}}tile_2_1) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_2_1, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_1, DMA : 0, %{{.*}}tile_2_2, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_1, DMA : 1, %{{.*}}tile_2_3, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_1, DMA : 2, %{{.*}}tile_3_3, DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @link1_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%{{.*}}tile_2_1) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb7)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb6
// CHECK:       aie.use_lock(@[[VAL_14]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_12_ACCESS_0:.*]] = aie.get_buffer @[[VAL_12]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_12_ACCESS_0]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_16]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_12_ACCESS_1:.*]] = aie.get_buffer @[[VAL_12]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_12_ACCESS_1]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_17]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(@[[VAL_18]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_12_ACCESS_2:.*]] = aie.get_buffer @[[VAL_12]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_12_ACCESS_2]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_19]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(@[[VAL_14]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_13_ACCESS_3:.*]] = aie.get_buffer @[[VAL_13]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_13_ACCESS_3]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(@[[VAL_16]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_13_ACCESS_4:.*]] = aie.get_buffer @[[VAL_13]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_13_ACCESS_4]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_17]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(@[[VAL_18]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_13_ACCESS_5:.*]] = aie.get_buffer @[[VAL_13]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_13_ACCESS_5]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_19]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb7:  // pred: ^bb0
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(MM2S, 0, ^bb8, ^bb10)
// CHECK:     ^bb8:  // 2 preds: ^bb7, ^bb9
// CHECK:       aie.use_lock(@[[VAL_15]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_12_ACCESS_6:.*]] = aie.get_buffer @[[VAL_12]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_12_ACCESS_6]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_14]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb9:  // pred: ^bb8
// CHECK:       aie.use_lock(@[[VAL_15]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_13_ACCESS_7:.*]] = aie.get_buffer @[[VAL_13]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_13_ACCESS_7]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_14]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb10:  // pred: ^bb7
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(MM2S, 1, ^bb11, ^bb13)
// CHECK:     ^bb11:  // 2 preds: ^bb10, ^bb12
// CHECK:       aie.use_lock(@[[VAL_17]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_12_ACCESS_8:.*]] = aie.get_buffer @[[VAL_12]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_12_ACCESS_8]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_16]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb12
// CHECK:     ^bb12:  // pred: ^bb11
// CHECK:       aie.use_lock(@[[VAL_17]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_13_ACCESS_9:.*]] = aie.get_buffer @[[VAL_13]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_13_ACCESS_9]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_16]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb13:  // pred: ^bb10
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(MM2S, 2, ^bb14, ^bb16)
// CHECK:     ^bb14:  // 2 preds: ^bb13, ^bb15
// CHECK:       aie.use_lock(@[[VAL_19]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_12_ACCESS_10:.*]] = aie.get_buffer @[[VAL_12]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_12_ACCESS_10]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_18]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb15
// CHECK:     ^bb15:  // pred: ^bb14
// CHECK:       aie.use_lock(@[[VAL_19]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_13_ACCESS_11:.*]] = aie.get_buffer @[[VAL_13]] : memref<48xi32>
// CHECK:       aie.dma_bd(%[[VAL_13_ACCESS_11]] : memref<48xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_18]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb14
// CHECK:     ^bb16:  // pred: ^bb13
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_12:.*]] = aie.get_buffer @[[VAL_8]] : memref<4x4xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_12]] : memref<4x4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_13:.*]] = aie.get_buffer @[[VAL_9]] : memref<4x4xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_13]] : memref<4x4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %[[CHECK_SSA_5:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_4_ACCESS_14:.*]] = aie.get_buffer @[[VAL_4]] : memref<20xi32>
// CHECK:       aie.dma_bd(%[[VAL_4_ACCESS_14]] : memref<20xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_5_ACCESS_15:.*]] = aie.get_buffer @[[VAL_5]] : memref<20xi32>
// CHECK:       aie.dma_bd(%[[VAL_5_ACCESS_15]] : memref<20xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %[[CHECK_SSA_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_16:.*]] = aie.get_buffer @[[VAL_0]] : memref<12xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_16]] : memref<12xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_1_ACCESS_17:.*]] = aie.get_buffer @[[VAL_1]] : memref<12xi32>
// CHECK:       aie.dma_bd(%[[VAL_1_ACCESS_17]] : memref<12xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
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

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

        aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
    }
}
