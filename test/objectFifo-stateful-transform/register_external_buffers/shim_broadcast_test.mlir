//===- shim_broadcast_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.tile(2, 3)
// CHECK-DAG:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:           aie.buffer @[[VAL_4:of_in_0_cons_buff_0]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_5:of_in_0_cons_buff_1]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_6:of_in_0_cons_prod_lock_0]](%[[VAL_1]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_7:of_in_0_cons_cons_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_8:of_in_1_cons_buff_0]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_9:of_in_1_cons_buff_1]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_10:of_in_1_cons_prod_lock_0]](%[[VAL_2]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_11:of_in_1_cons_cons_lock_0]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_12:of_in_2_cons_buff_0]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_13:of_in_2_cons_buff_1]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_14:of_in_2_cons_prod_lock_0]](%[[VAL_3]]) {init = 2 : i32}
// CHECK-DAG:           aie.lock @[[VAL_15:of_in_2_cons_cons_lock_0]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_16:of_in_prod_lock_0]](%[[VAL_0]]) {init = 1 : i32}
// CHECK-DAG:           aie.lock @[[VAL_17:of_in_cons_lock_0]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK-DAG:           aie.external_buffer @[[VAL_18:ext_buffer_in]] : memref<64xi32>
// CHECK-DAG:           aie.shim_dma_allocation @of_in_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_19:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(@[[VAL_17]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_18_ACCESS_0:.*]] = aie.get_buffer @[[VAL_18]] : memref<64xi32>
// CHECK:             aie.dma_bd(%[[VAL_18_ACCESS_0]] : memref<64xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_16]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_4_ACCESS_1:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_4_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_6]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_5_ACCESS_2:.*]] = aie.get_buffer @[[VAL_5]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_5_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_8_ACCESS_3:.*]] = aie.get_buffer @[[VAL_8]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_8_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_10]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_9_ACCESS_4:.*]] = aie.get_buffer @[[VAL_9]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_9_ACCESS_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_14]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_12_ACCESS_5:.*]] = aie.get_buffer @[[VAL_12]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_12_ACCESS_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_14]], AcquireGreaterEqual, %{{.*}})
// CHECK:             %[[VAL_13_ACCESS_6:.*]] = aie.get_buffer @[[VAL_13]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_13_ACCESS_6]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @shim_broadcast {
   aie.device(xcve2302) {
      %tile20 = aie.tile(2, 0)
      %tile22 = aie.tile(2, 2)
      %tile23 = aie.tile(2, 3)
      %tile33 = aie.tile(3, 3)

      aie.objectfifo @of_in (%tile20, {%tile22, %tile23, %tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      aie.external_buffer @ext_buffer_in: memref<64xi32>
      aie.objectfifo.register_external_buffers @of_in (%tile20, [@ext_buffer_in])
   }
}
