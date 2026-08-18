//===- base_test_AIE1.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: July 26th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_2"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of0_lock_3"}
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_lock_1"}
// CHECK:           %[[VAL_13:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_13]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_13]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_13]]) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_13]], DMA : 0)
// CHECK:           %[[VAL_18:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.mem(%[[VAL_13]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_16]], Acquire, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_16]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_17]], Acquire, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
