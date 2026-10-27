//===- repeat_count_test.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @repeatCount {
// CHECK:   aie.device(npu1) {
// CHECK-DAG:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK-DAG:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK-DAG:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK-DAG:     aie.buffer @[[VAL_0:of1_cons_buff_0]](%{{.*}}tile_1_3) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_1:of1_cons_prod_lock_0]](%{{.*}}tile_1_3) {init = 1 : i32}
// CHECK-DAG:     aie.lock @[[VAL_2:of1_cons_cons_lock_0]](%{{.*}}tile_1_3) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_3:of1_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_4:of1_prod_lock_0]](%{{.*}}tile_1_2) {init = 3 : i32}
// CHECK-DAG:     aie.lock @[[VAL_5:of1_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_6:of0_cons_buff_0]](%{{.*}}tile_1_2) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_7:of0_cons_prod_lock_0]](%{{.*}}tile_1_2) {init = 1 : i32}
// CHECK-DAG:     aie.lock @[[VAL_8:of0_cons_cons_lock_0]](%{{.*}}tile_1_2) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[VAL_9:of0_buff_0]](%{{.*}}tile_1_1) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_10:of0_prod_lock_0]](%{{.*}}tile_1_1) {init = 1 : i32}
// CHECK-DAG:     aie.lock @[[VAL_11:of0_cons_lock_0]](%{{.*}}tile_1_1) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_2, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
// CHECK:     %core_1_2 = aie.core(%{{.*}}tile_1_2) {
// CHECK-DAG:       %c0 = arith.constant 0 : index
// CHECK-DAG:       %c1 = arith.constant 1 : index
// CHECK-DAG:       %c12 = arith.constant 12 : index
// CHECK:       scf.for %arg0 = %c0 to %c12 step %c1 {
// CHECK:         aie.use_lock(@[[VAL_8]], AcquireGreaterEqual, %{{.*}})
// CHECK:         aie.use_lock(@[[VAL_4]], AcquireGreaterEqual, %{{.*}})
// CHECK:         aie.use_lock(@[[VAL_5]], Release, %{{.*}})
// CHECK:         aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:       }
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(@[[VAL_11]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_9_ACCESS_0:.*]] = aie.get_buffer @[[VAL_9]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_9_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %[[CHECK_SSA_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(@[[VAL_7]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_6_ACCESS_1:.*]] = aie.get_buffer @[[VAL_6]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_6_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_8]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %[[CHECK_SSA_2:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 2)
// CHECK:     ^bb3:
// CHECK:       aie.use_lock(@[[VAL_5]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_3_ACCESS_2:.*]] = aie.get_buffer @[[VAL_3]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_3_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%{{.*}}tile_1_3) {
// CHECK:       %[[CHECK_SSA_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(@[[VAL_1]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[VAL_0_ACCESS_3:.*]] = aie.get_buffer @[[VAL_0]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_0_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_2]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @repeatCount {
 aie.device(npu1) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile11, {%tile12}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>

    %core33 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %height = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %height step %c1 {
         %subview0_obj0 = aie.objectfifo.acquire @of0 (Consume, 1) : memref<16xi32>
         %subview1_obj0 = aie.objectfifo.acquire @of1 (Produce, 1) : memref<16xi32>
         aie.objectfifo.release @of1 (Produce, 1)
         aie.objectfifo.release @of0 (Consume, 1)
      }

      aie.end
   }
 }
}
