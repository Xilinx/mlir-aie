//===- link_test_AIE1.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK-DAG:           aie.buffer @[[VAL_3:of1_cons_buff_0]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_4:of1_cons_buff_1]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_5:of1_cons_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_6:of1_cons_lock_1]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_7:of1_lock_0]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK-DAG:           aie.external_buffer @[[VAL_8:ext_buff_in]] : memref<16xi32>
// CHECK-DAG:           aie.shim_dma_allocation @of1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_9:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(@[[VAL_7]], Acquire, %{{.*}})
// CHECK:             %[[VAL_8_ACCESS_0:.*]] = aie.get_buffer @[[VAL_8]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_8_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_5]], Acquire, %{{.*}})
// CHECK:             %[[VAL_3_ACCESS_1:.*]] = aie.get_buffer @[[VAL_3]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_3_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_5]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_6]], Acquire, %{{.*}})
// CHECK:             %[[VAL_4_ACCESS_2:.*]] = aie.get_buffer @[[VAL_4]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_4_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_6]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_AIE1 {
    aie.device(xcvc1902) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ([] [])

        aie.external_buffer @ext_buff_in : memref<16xi32>
        aie.objectfifo.register_external_buffers @of1 (%tile20, [@ext_buff_in])
    }
}
