//===- memtile_repeat_count_test.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @memtileRepeat {
// CHECK:   aie.device(npu1) {
// CHECK-DAG:     %{{.*}}tile_1_0 = aie.tile(1, 0)
// CHECK-DAG:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_3_3) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK-DAG:     %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_3_3) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK-DAG:     %[[VAL_4:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_5:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_1_2) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK-DAG:     %[[VAL_7:.*]] = aie.lock(%{{.*}}tile_1_2) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK-DAG:     %[[VAL_8:.*]] = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of0_cons_buff_0"} : memref<32xi32>
// CHECK-DAG:     %[[VAL_9:.*]] = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of0_cons_buff_1"} : memref<32xi32>
// CHECK-DAG:     %[[VAL_10:.*]] = aie.lock(%{{.*}}tile_1_1) {init = 6 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK-DAG:     %[[VAL_11:.*]] = aie.lock(%{{.*}}tile_1_1) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK-DAG:     %[[VAL_12:.*]] = aie.lock(%{{.*}}tile_1_1) {init = 6 : i32, sym_name = "of0_cons_prod_lock_1"}
// CHECK-DAG:     %[[VAL_13:.*]] = aie.lock(%{{.*}}tile_1_1) {init = 0 : i32, sym_name = "of0_cons_cons_lock_1"}
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_0, DMA : 0, %{{.*}}tile_1_1, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_1, DMA : 1, %{{.*}}tile_3_3, DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @of0_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_13]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_11]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_13]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb12)
// CHECK:     ^bb6:  // 2 preds: ^bb5, ^bb11
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb9:  // pred: ^bb8
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb12:  // pred: ^bb5
// CHECK:       %2 = aie.dma_start(MM2S, 1, ^bb13, ^bb19)
// CHECK:     ^bb13:  // 2 preds: ^bb12, ^bb18
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb14
// CHECK:     ^bb14:  // pred: ^bb13
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb15
// CHECK:     ^bb15:  // pred: ^bb14
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb16
// CHECK:     ^bb16:  // pred: ^bb15
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb17
// CHECK:     ^bb17:  // pred: ^bb16
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb18
// CHECK:     ^bb18:  // pred: ^bb17
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<32xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb13
// CHECK:     ^bb19:  // pred: ^bb12
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_7]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile10, {%tile11}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of1 (%tile11, {%tile12}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile11, {%tile33}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of0] -> [@of1, @of2] ([] [0, 16])
 }
}
