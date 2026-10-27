//===- nd_dma_multiple_consumers_AIE2.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @ndDMAObjFifoAIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %[[tile_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:     %[[tile_1_3:.*]] = aie.tile(1, 3)
// CHECK-DAG:     %[[tile_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:     %[[tile_2_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:     %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:     aie.buffer @[[of3_cons_buff_0:of3_cons_buff_0]](%[[tile_2_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of3_cons_buff_1:of3_cons_buff_1]](%[[tile_2_3]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of3_cons_prod_lock:of3_cons_prod_lock_0]](%[[tile_2_3]]) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[of3_cons_cons_lock:of3_cons_cons_lock_0]](%[[tile_2_3]]) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[of3_buff_0:of3_buff_0]](%[[tile_2_2]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of3_buff_1:of3_buff_1]](%[[tile_2_2]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of3_prod_lock:of3_prod_lock_0]](%[[tile_2_2]]) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[of3_cons_lock:of3_cons_lock_0]](%[[tile_2_2]]) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[of1_cons_buff_0:of1_cons_buff_0]](%[[tile_3_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of1_cons_buff_1:of1_cons_buff_1]](%[[tile_3_3]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of1_cons_prod_lock:of1_cons_prod_lock_0]](%[[tile_3_3]]) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[of1_cons_cons_lock:of1_cons_cons_lock_0]](%[[tile_3_3]]) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[of1_buff_0:of1_buff_0]](%[[tile_1_2]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of1_buff_1:of1_buff_1]](%[[tile_1_2]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of1_prod_lock:of1_prod_lock_0]](%[[tile_1_2]]) {init = 2 : i32}
// CHECK-DAG:     aie.lock @[[of1_cons_lock:of1_cons_lock_0]](%[[tile_1_2]]) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[of0_0_cons_buff_0:of0_0_cons_buff_0]](%[[tile_1_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_0_cons_buff_1:of0_0_cons_buff_1]](%[[tile_1_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_0_cons_buff_2:of0_0_cons_buff_2]](%[[tile_1_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_0_cons_buff_3:of0_0_cons_buff_3]](%[[tile_1_3]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of0_0_cons_prod_lock:of0_0_cons_prod_lock_0]](%[[tile_1_3]]) {init = 4 : i32}
// CHECK-DAG:     aie.lock @[[of0_0_cons_cons_lock:of0_0_cons_cons_lock_0]](%[[tile_1_3]]) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[of0_1_cons_buff_0:of0_1_cons_buff_0]](%[[tile_3_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_1_cons_buff_1:of0_1_cons_buff_1]](%[[tile_3_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_1_cons_buff_2:of0_1_cons_buff_2]](%[[tile_3_3]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_1_cons_buff_3:of0_1_cons_buff_3]](%[[tile_3_3]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of0_1_cons_prod_lock:of0_1_cons_prod_lock_0]](%[[tile_3_3]]) {init = 4 : i32}
// CHECK-DAG:     aie.lock @[[of0_1_cons_cons_lock:of0_1_cons_cons_lock_0]](%[[tile_3_3]]) {init = 0 : i32}
// CHECK-DAG:     aie.buffer @[[of0_buff_0:of0_buff_0]](%[[tile_1_2]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_buff_1:of0_buff_1]](%[[tile_1_2]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_buff_2:of0_buff_2]](%[[tile_1_2]]) : memref<256xi32>
// CHECK-DAG:     aie.buffer @[[of0_buff_3:of0_buff_3]](%[[tile_1_2]]) : memref<256xi32>
// CHECK-DAG:     aie.lock @[[of0_prod_lock:of0_prod_lock_0]](%[[tile_1_2]]) {init = 4 : i32}
// CHECK-DAG:     aie.lock @[[of0_cons_lock:of0_cons_lock_0]](%[[tile_1_2]]) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%[[tile_1_2]], DMA : 1, %[[tile_3_3]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[tile_1_2]], DMA : 1, %[[tile_1_3]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[tile_1_2]], DMA : 0, %[[tile_3_3]], DMA : 1)
// CHECK-DAG:     aie.flow(%[[tile_2_2]], DMA : 0, %[[tile_2_3]], DMA : 0)
// CHECK:     %[[VAL_39:.*]] = aie.mem(%[[tile_1_2]]) {
// CHECK:       %[[VAL_45:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:       aie.use_lock(@[[of1_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of1_buff_0_ACCESS_0:.*]] = aie.get_buffer @[[of1_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of1_buff_0_ACCESS_0]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of1_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:       aie.use_lock(@[[of1_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of1_buff_1_ACCESS_1:.*]] = aie.get_buffer @[[of1_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of1_buff_1_ACCESS_1]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of1_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:       %[[VAL_44:.*]] = aie.dma_start(MM2S, 1, ^bb4, ^bb8)
// CHECK:           ^bb4:
// CHECK:       aie.use_lock(@[[of0_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_buff_0_ACCESS_2:.*]] = aie.get_buffer @[[of0_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_buff_0_ACCESS_2]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:       aie.use_lock(@[[of0_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_buff_1_ACCESS_3:.*]] = aie.get_buffer @[[of0_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_buff_1_ACCESS_3]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:       aie.use_lock(@[[of0_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_buff_2_ACCESS_4:.*]] = aie.get_buffer @[[of0_buff_2]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_buff_2_ACCESS_4]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:       aie.use_lock(@[[of0_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_buff_3_ACCESS_5:.*]] = aie.get_buffer @[[of0_buff_3]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_buff_3_ACCESS_5]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb8:
// CHECK:             aie.end
// CHECK:           }
// CHECK:     %[[VAL_40:.*]] = aie.mem(%[[tile_1_3]]) {
// CHECK:       %[[VAL_44:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(@[[of0_0_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_0_cons_buff_0_ACCESS_6:.*]] = aie.get_buffer @[[of0_0_cons_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_0_cons_buff_0_ACCESS_6]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_0_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[of0_0_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_0_cons_buff_1_ACCESS_7:.*]] = aie.get_buffer @[[of0_0_cons_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_0_cons_buff_1_ACCESS_7]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_0_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(@[[of0_0_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_0_cons_buff_2_ACCESS_8:.*]] = aie.get_buffer @[[of0_0_cons_buff_2]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_0_cons_buff_2_ACCESS_8]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_0_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(@[[of0_0_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_0_cons_buff_3_ACCESS_9:.*]] = aie.get_buffer @[[of0_0_cons_buff_3]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_0_cons_buff_3_ACCESS_9]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_0_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:     %[[VAL_41:.*]] = aie.mem(%[[tile_3_3]]) {
// CHECK:       %[[VAL_44:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(@[[of0_1_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_1_cons_buff_0_ACCESS_10:.*]] = aie.get_buffer @[[of0_1_cons_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_1_cons_buff_0_ACCESS_10]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_1_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[of0_1_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_1_cons_buff_1_ACCESS_11:.*]] = aie.get_buffer @[[of0_1_cons_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_1_cons_buff_1_ACCESS_11]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_1_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(@[[of0_1_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_1_cons_buff_2_ACCESS_12:.*]] = aie.get_buffer @[[of0_1_cons_buff_2]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_1_cons_buff_2_ACCESS_12]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_1_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(@[[of0_1_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of0_1_cons_buff_3_ACCESS_13:.*]] = aie.get_buffer @[[of0_1_cons_buff_3]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of0_1_cons_buff_3_ACCESS_13]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of0_1_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:       %[[VAL_45:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:       aie.use_lock(@[[of1_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of1_cons_buff_0_ACCESS_14:.*]] = aie.get_buffer @[[of1_cons_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of1_cons_buff_0_ACCESS_14]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of1_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(@[[of1_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of1_cons_buff_1_ACCESS_15:.*]] = aie.get_buffer @[[of1_cons_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of1_cons_buff_1_ACCESS_15]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of1_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK:     %[[VAL_42:.*]] = aie.mem(%[[tile_2_2]]) {
// CHECK:       %[[VAL_44:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[of3_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of3_buff_0_ACCESS_16:.*]] = aie.get_buffer @[[of3_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of3_buff_0_ACCESS_16]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of3_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[of3_cons_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of3_buff_1_ACCESS_17:.*]] = aie.get_buffer @[[of3_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of3_buff_1_ACCESS_17]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of3_prod_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:     %[[VAL_43:.*]] = aie.mem(%[[tile_2_3]]) {
// CHECK:       %[[VAL_44:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[of3_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of3_cons_buff_0_ACCESS_18:.*]] = aie.get_buffer @[[of3_cons_buff_0]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of3_cons_buff_0_ACCESS_18]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of3_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[of3_cons_prod_lock]], AcquireGreaterEqual, %{{.*}})
// CHECK:       %[[of3_cons_buff_1_ACCESS_19:.*]] = aie.get_buffer @[[of3_cons_buff_1]] : memref<256xi32>
// CHECK:       aie.dma_bd(%[[of3_cons_buff_1_ACCESS_19]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[of3_cons_cons_lock]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK: }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12 dimensionsToStream [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>], // transpose
                         {%tile13 dimensionsFromStream [<size = 1, stride = 1>],
                          %tile33 dimensionsFromStream [<size = 3, stride = 4>]},
                         4 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile12 dimensionsToStream [<size = 128, stride = 2>], {%tile33},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of3 (%tile22, {%tile23 dimensionsFromStream [<size = 9, stride = 9>]},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>
 }
}
