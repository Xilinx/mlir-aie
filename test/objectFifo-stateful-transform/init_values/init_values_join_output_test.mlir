//===- init_values_join_output_test.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @init_join_output {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK-DAG:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK-DAG: aie.buffer @[[OF2_BUFF_0:of2_buff_0]](%{{.*}}tile_1_1) : memref<4xi32> = dense<[0, 1, 2, 3]>
// CHECK-DAG: aie.buffer @[[OF2_BUFF_1:of2_buff_1]](%{{.*}}tile_1_1) : memref<4xi32> = dense<[4, 5, 6, 7]>
// CHECK-DAG: aie.lock @[[OF2_PROD_LOCK_0:of2_prod_lock_0]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG: aie.lock @[[OF2_CONS_LOCK_0:of2_cons_lock_0]](%{{.*}}tile_1_1) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[OF2_PROD_LOCK_1:of2_prod_lock_1]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG: aie.lock @[[OF2_CONS_LOCK_1:of2_cons_lock_1]](%{{.*}}tile_1_1) {init = 2 : i32}
// CHECK-DAG: aie.buffer @[[OF1_BUFF_0:of1_buff_0]](%{{.*}}tile_2_3) : memref<2xi32>
// CHECK-DAG: aie.buffer @[[OF1_BUFF_1:of1_buff_1]](%{{.*}}tile_2_3) : memref<2xi32>
// CHECK-DAG: aie.lock @[[OF1_PROD_LOCK:of1_prod_lock_0]](%{{.*}}tile_2_3) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[OF1_CONS_LOCK:of1_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG: aie.buffer @[[OF0_BUFF_0:of0_buff_0]](%{{.*}}tile_1_2) : memref<2xi32>
// CHECK-DAG: aie.buffer @[[OF0_BUFF_1:of0_buff_1]](%{{.*}}tile_1_2) : memref<2xi32>
// CHECK-DAG: aie.lock @[[OF0_PROD_LOCK:of0_prod_lock_0]](%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK-DAG: aie.lock @[[OF0_CONS_LOCK:of0_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG: aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_2_3, DMA : 0, %{{.*}}tile_1_1, DMA : 1)
// CHECK-DAG: aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_0, DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @of2_shim_alloc(%[[VAL_0]], S2MM, 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF0_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF0_BUFF_0_ACCESS_0:.*]] = aie.get_buffer @[[OF0_BUFF_0]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[OF0_BUFF_0_ACCESS_0]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF0_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[OF0_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF0_BUFF_1_ACCESS_1:.*]] = aie.get_buffer @[[OF0_BUFF_1]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[OF0_BUFF_1_ACCESS_1]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF0_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_0_ACCESS_2:.*]] = aie.get_buffer @[[OF2_BUFF_0]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0_ACCESS_2]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_1_ACCESS_3:.*]] = aie.get_buffer @[[OF2_BUFF_1]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1_ACCESS_3]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_0_ACCESS_4:.*]] = aie.get_buffer @[[OF2_BUFF_0]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0_ACCESS_4]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_1_ACCESS_5:.*]] = aie.get_buffer @[[OF2_BUFF_1]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1_ACCESS_5]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb11)
// CHECK:     ^bb7:
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_0_ACCESS_6:.*]] = aie.get_buffer @[[OF2_BUFF_0]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0_ACCESS_6]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_0_ACCESS_7:.*]] = aie.get_buffer @[[OF2_BUFF_0]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0_ACCESS_7]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb9:
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_0]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_1_ACCESS_8:.*]] = aie.get_buffer @[[OF2_BUFF_1]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1_ACCESS_8]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_0]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(@[[OF2_CONS_LOCK_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF2_BUFF_1_ACCESS_9:.*]] = aie.get_buffer @[[OF2_BUFF_1]] : memref<4xi32>
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1_ACCESS_9]] : memref<4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF2_PROD_LOCK_1]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb11:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(@[[OF1_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF1_BUFF_0_ACCESS_10:.*]] = aie.get_buffer @[[OF1_BUFF_0]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[OF1_BUFF_0_ACCESS_10]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF1_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(@[[OF1_CONS_LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[OF1_BUFF_1_ACCESS_11:.*]] = aie.get_buffer @[[OF1_BUFF_1]] : memref<2xi32>
// CHECK:       aie.dma_bd(%[[OF1_BUFF_1_ACCESS_11]] : memref<2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[OF1_PROD_LOCK]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @init_join_output {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    aie.objectfifo @of1 (%tile23, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    aie.objectfifo @of2 (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>,
                                                                                          dense<[4, 5, 6, 7]> : memref<4xi32>]

    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 2] [])
 }
}
