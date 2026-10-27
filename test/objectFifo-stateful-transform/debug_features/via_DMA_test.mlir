//===- via_DMA_test.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @viaDMA {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:of_stream_cons_buff_0]](%{{.*}}tile_1_3) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_1:of_stream_cons_buff_1]](%{{.*}}tile_1_3) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_2:of_stream_cons_prod_lock_0]](%{{.*}}tile_1_3) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_3:of_stream_cons_cons_lock_0]](%{{.*}}tile_1_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_4:of_stream_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_5:of_stream_buff_1]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_6:of_stream_prod_lock_0]](%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_7:of_stream_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_8:of_shared_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_9:of_shared_buff_1]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_10:of_shared_prod_lock_0]](%{{.*}}tile_1_2) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[VAL_11:of_shared_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_7]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_4_ACCESS_0:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_4_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_7]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_5_ACCESS_1:.*]] = aie.get_buffer @[[VAL_5]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_5_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%{{.*}}tile_1_3) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_2:.*]] = aie.get_buffer @[[VAL_0]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_2]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_1_ACCESS_3:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_1_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_3]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @viaDMA {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_shared (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_stream (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>
 }
}
