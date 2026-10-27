//===- init_values_test.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @init {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:of0_cons_buff_0]](%{{.*}}tile_2_3) : memref<2x2xi32>
// CHECK-DAG:     aie.buffer @[[VAL_1:of0_cons_buff_1]](%{{.*}}tile_2_3) : memref<2x2xi32>
// CHECK-DAG:     aie.buffer @[[VAL_2:of0_cons_buff_2]](%{{.*}}tile_2_3) : memref<2x2xi32>
// CHECK-DAG:     aie.lock @[[VAL_3:of0_cons_prod_lock_0]](%{{.*}}tile_2_3) {init = 3 : i32}
// CHECK-DAG:     aie.lock @[[VAL_4:of0_cons_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_5:of0_buff_0]](%{{.*}}tile_1_2) : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK-DAG:     aie.buffer @[[VAL_6:of0_buff_1]](%{{.*}}tile_1_2) : memref<2x2xi32> = dense<{{\[}}[4, 5], [6, 7]]>
// CHECK-DAG:     aie.buffer @[[VAL_7:of0_buff_2]](%{{.*}}tile_1_2) : memref<2x2xi32> = dense<{{\[}}[8, 9], [10, 11]]>
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_2_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK-NOT:   aie.use_lock
// CHECK:       %[[VAL_5_ACCESS_0:.*]] = aie.get_buffer @[[VAL_5]] : memref<2x2xi32>
// CHECK:       aie.dma_bd(%[[VAL_5_ACCESS_0]] : memref<2x2xi32> offset = {{.*}} len = {{.*}})
// CHECK-NOT:   aie.use_lock
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK-NOT:   aie.use_lock
// CHECK:       %[[VAL_6_ACCESS_1:.*]] = aie.get_buffer @[[VAL_6]] : memref<2x2xi32>
// CHECK:       aie.dma_bd(%[[VAL_6_ACCESS_1]] : memref<2x2xi32> offset = {{.*}} len = {{.*}})
// CHECK-NOT:   aie.use_lock
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK-NOT:   aie.use_lock
// CHECK:       %[[VAL_7_ACCESS_2:.*]] = aie.get_buffer @[[VAL_7]] : memref<2x2xi32>
// CHECK:       aie.dma_bd(%[[VAL_7_ACCESS_2]] : memref<2x2xi32> offset = {{.*}} len = {{.*}})
// CHECK-NOT:   aie.use_lock
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(@[[VAL_3]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_3:.*]] = aie.get_buffer @[[VAL_0]] : memref<2x2xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_3]] : memref<2x2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_3]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_1_ACCESS_4:.*]] = aie.get_buffer @[[VAL_1]] : memref<2x2xi32>
// CHECK:       aie.dma_bd(%[[VAL_1_ACCESS_4]] : memref<2x2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(@[[VAL_3]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_2_ACCESS_5:.*]] = aie.get_buffer @[[VAL_2]] : memref<2x2xi32>
// CHECK:       aie.dma_bd(%[[VAL_2_ACCESS_5]] : memref<2x2xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @init {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile23}, 3 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>,
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>,
                                                                                            dense<[[8, 9], [10, 11]]> : memref<2x2xi32>]
 }
}
