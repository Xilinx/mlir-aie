//===- base_test_AIE1.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: July 26th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK: module @elementGenerationAIE1 {
// CHECK:   aie.device(xcvc1902) {
// CHECK-DAG:     %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:     %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK-DAG:     %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK-DAG:     %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK-DAG:     %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_9:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK-DAG:     %[[VAL_10:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_lock_1"}
// CHECK-DAG:     %[[VAL_11:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_12:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_13:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_2"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_14:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_3"} : memref<16xi32>
// CHECK-DAG:     %[[VAL_15:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_0"}
// CHECK-DAG:     %[[VAL_16:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_1"}
// CHECK-DAG:     %[[VAL_17:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_2"}
// CHECK-DAG:     %[[VAL_18:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_3"}
// CHECK-DAG:     aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:     %[[VAL_19:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:       aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_9]], Acquire, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_9]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_10]], Acquire, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_10]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[VAL_20:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:       aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_5]], Acquire, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_5]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_6]], Acquire, %{{.*}})
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:       aie.use_lock(%[[VAL_6]], Release, %{{.*}})
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @elementGenerationAIE1 {
   aie.device(xcvc1902) {
      %tile12 = aie.tile(1, 2)
      %tile13 = aie.tile(1, 3)
      %tile33 = aie.tile(3, 3)

      // In the shared memory case, the number of elements does not change.
      aie.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

      // In the non-adjacent memory case, the number of elements depends on the max amount acquired by
      // the processes running on each core (here nothing is specified so it cannot be derived).
      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
