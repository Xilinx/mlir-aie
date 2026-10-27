//===- init_values_distribute_input_test.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @init_distribute_input {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK-DAG:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:of2_cons_buff_0]](%{{.*}}tile_2_3) : memref<2xi32>
// CHECK-DAG:     aie.buffer @[[VAL_1:of2_cons_buff_1]](%{{.*}}tile_2_3) : memref<2xi32>
// CHECK-DAG:     aie.lock @[[VAL_2:of2_cons_prod_lock_0]](%{{.*}}tile_2_3) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_3:of2_cons_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_4:of1_cons_buff_0]](%{{.*}}tile_1_2) : memref<2xi32>
// CHECK-DAG:     aie.buffer @[[VAL_5:of1_cons_buff_1]](%{{.*}}tile_1_2) : memref<2xi32>
// CHECK-DAG:     aie.lock @[[VAL_6:of1_cons_prod_lock_0]](%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_7:of1_cons_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_8:of0_cons_buff_0]](%{{.*}}tile_1_1) : memref<4xi32>
// CHECK-DAG:     aie.buffer @[[VAL_9:of0_cons_buff_1]](%{{.*}}tile_1_1) : memref<4xi32>
// CHECK-DAG:     aie.lock @[[VAL_10:of0_cons_prod_lock_0]](%{{.*}}tile_1_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_11:of0_cons_cons_lock_0]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG:     aie.lock @[[VAL_12:of0_cons_prod_lock_1]](%{{.*}}tile_1_1) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_13:of0_cons_cons_lock_1]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_14:of0_buff_0]](%{{.*}}tile_1_3) : memref<4xi32> = dense<[0, 1, 2, 3]>
// CHECK-DAG:     aie.buffer @[[VAL_15:of0_buff_1]](%{{.*}}tile_1_3) : memref<4xi32> = dense<[4, 5, 6, 7]>
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_3, DMA : 0, %{{.*}}tile_1_1, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_1, DMA : 1, %{{.*}}tile_2_3, DMA : 0)
// CHECK:     %mem_1_3 = aie.mem(%{{.*}}tile_1_3) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK-NOT:   aie.use_lock
// CHECK:       %[[VAL_14_ACCESS_0:.*]] = aie.get_buffer @[[VAL_14]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_14_ACCESS_0]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK-NOT:   aie.use_lock
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK-NOT:   aie.use_lock
// CHECK:       %[[VAL_15_ACCESS_1:.*]] = aie.get_buffer @[[VAL_15]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_15_ACCESS_1]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK-NOT:   aie.use_lock
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_2:.*]] = aie.get_buffer @[[VAL_8]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_2]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_12]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_3:.*]] = aie.get_buffer @[[VAL_8]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_3]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_4:.*]] = aie.get_buffer @[[VAL_9]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_4]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(@[[VAL_12]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_5:.*]] = aie.get_buffer @[[VAL_9]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_5]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:     ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:       aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_6:.*]] = aie.get_buffer @[[VAL_8]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_6]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_7:.*]] = aie.get_buffer @[[VAL_9]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_7]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb8:  // pred: ^bb5
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
// CHECK:     ^bb9:  // 2 preds: ^bb8, ^bb10
// CHECK:       aie.use_lock(@[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_8_ACCESS_8:.*]] = aie.get_buffer @[[VAL_8]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_8_ACCESS_8]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(@[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_9:.*]] = aie.get_buffer @[[VAL_9]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_9]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb11:  // pred: ^bb8
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_4_ACCESS_10:.*]] = aie.get_buffer @[[VAL_4]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[VAL_4_ACCESS_10]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_5_ACCESS_11:.*]] = aie.get_buffer @[[VAL_5]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[VAL_5_ACCESS_11]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %[[CHECK_SSA_5:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_12:.*]] = aie.get_buffer @[[VAL_0]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_12]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_1_ACCESS_13:.*]] = aie.get_buffer @[[VAL_1]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[VAL_1_ACCESS_13]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @init_distribute_input {
 aie.device(xcve2302) {
    %tile13 = aie.tile(1, 3)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile13, {%tile11}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>,
                                                                                          dense<[4, 5, 6, 7]> : memref<4xi32>]
    aie.objectfifo @of1 (%tile11, {%tile12}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    aie.objectfifo @of2 (%tile11, {%tile23}, 2 : i32) : !aie.objectfifo<memref<2xi32>>

    aie.objectfifo.link [@of0] -> [@of1, @of2] ([] [0, 2])
 }
}
