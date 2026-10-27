//===- disable_synchronization_test_distribute.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @disable_sync {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK-DAG:     %{{.*}}tile_2_1 = aie.tile(2, 1)
// CHECK-DAG:     %{{.*}}tile_2_2 = aie.tile(2, 2)
// CHECK-DAG:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:link3_cons_buff_0]](%{{.*}}tile_2_3) : memref<20xi32>
// CHECK-DAG:     aie.lock @[[VAL_1:link3_cons_prod_lock_0]](%{{.*}}tile_2_3) {init = 1 : i32}
// CHECK-DAG:     aie.lock @[[VAL_2:link3_cons_cons_lock_0]](%{{.*}}tile_2_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_3:link2_cons_buff_0]](%{{.*}}tile_2_2) : memref<4x4xi32>
// CHECK-DAG:     aie.lock @[[VAL_4:link2_cons_prod_lock_0]](%{{.*}}tile_2_2) {init = 1 : i32}
// CHECK-DAG:     aie.lock @[[VAL_5:link2_cons_cons_lock_0]](%{{.*}}tile_2_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_6:link1_cons_buff_0]](%{{.*}}tile_2_1) : memref<36xi32>
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_0, DMA : 0, %{{.*}}tile_2_1, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_1, DMA : 0, %{{.*}}tile_2_2, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_2_1, DMA : 1, %{{.*}}tile_2_3, DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @link1_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%{{.*}}tile_2_1) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       %[[VAL_6_ACCESS_0:.*]] = aie.get_buffer @[[VAL_6]] : memref<36xi32>
// CHECK:       aie.dma_bd(%[[VAL_6_ACCESS_0]] : memref<36xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       %[[VAL_6_ACCESS_1:.*]] = aie.get_buffer @[[VAL_6]] : memref<36xi32>
// CHECK:       aie.dma_bd(%[[VAL_6_ACCESS_1]] : memref<36xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb5)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:       %[[VAL_6_ACCESS_2:.*]] = aie.get_buffer @[[VAL_6]] : memref<36xi32>
// CHECK:       aie.dma_bd(%[[VAL_6_ACCESS_2]] : memref<36xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb5:  // pred: ^bb3
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb7)
// CHECK:     ^bb6:  // 2 preds: ^bb5, ^bb6
// CHECK:       %[[VAL_6_ACCESS_3:.*]] = aie.get_buffer @[[VAL_6]] : memref<36xi32>
// CHECK:       aie.dma_bd(%[[VAL_6_ACCESS_3]] : memref<36xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb7:  // pred: ^bb5
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(@[[VAL_4]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_3_ACCESS_4:.*]] = aie.get_buffer @[[VAL_3]] : memref<4x4xi32>
// CHECK:       aie.dma_bd(%[[VAL_3_ACCESS_4]] : memref<4x4xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_5]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %[[CHECK_SSA_4:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(@[[VAL_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_5:.*]] = aie.get_buffer @[[VAL_0]] : memref<20xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_5]] : memref<20xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_2]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @disable_sync {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @link1 (%tile20, {%tile21}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<36xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile21, {%tile23}, 1 : i32) : !aie.objectfifo<memref<20xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3] ([][0, 16])
 }
}
