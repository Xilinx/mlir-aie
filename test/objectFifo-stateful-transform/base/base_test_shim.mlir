//===- base_test_shim.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @elementGenerationShim {
// CHECK:   aie.device(xcvc1902) {
// CHECK-DAG:     %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK-DAG:     %[[VAL_1:.*]] = aie.tile(1, 2)
// CHECK-DAG:     aie.buffer @[[VAL_2:of1_cons_buff_0]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:     aie.buffer @[[VAL_3:of1_cons_buff_1]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:     aie.lock @[[VAL_4:of1_cons_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:     aie.lock @[[VAL_5:of1_cons_lock_1]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:     aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK-DAG:     aie.shim_dma_allocation @of1_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:     %[[VAL_6:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:       %[[CHECK_SSA_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(@[[VAL_4]], Acquire, %{{.*}})
// CHECK:       %[[VAL_2_ACCESS_0:.*]] = aie.get_buffer @[[VAL_2]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_2_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(@[[VAL_5]], Acquire, %{{.*}})
// CHECK:       %[[VAL_3_ACCESS_1:.*]] = aie.get_buffer @[[VAL_3]] : memref<16xi32>
// CHECK:       aie.dma_bd(%[[VAL_3_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(@[[VAL_5]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @elementGenerationShim {
   aie.device(xcvc1902) {
      %tile10 = aie.tile(1, 0)
      %tile12 = aie.tile(1, 2)

      aie.objectfifo @of1 (%tile10, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
