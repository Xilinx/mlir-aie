//===- lock_analysis_test.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_5]]) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "test_buff"} : memref<16xi32>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_13:.*]] = aie.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "test_prod_lock"}
// CHECK:             %[[VAL_14:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "test_cons_lock"}
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_14]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_8]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_18]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @lockAnalysis {
   aie.device(xcve2302) {
      %tile12 = aie.tile(1, 2)
      %tile33 = aie.tile(3, 3)

      %test_buff = aie.buffer(%tile12) {sym_name = "test_buff"} : memref<16xi32>

      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      %mem_1_2 = aie.mem(%tile12) {
         %test_prod_lock = aie.lock(%tile12, 0) {init = 1 : i32, sym_name = "test_prod_lock"}
         %test_cons_lock = aie.lock(%tile12, 1) {init = 0 : i32, sym_name = "test_cons_lock"}
         %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
         ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(%test_prod_lock, AcquireGreaterEqual, %c1_ul1)
            aie.dma_bd(%test_buff : memref<16xi32> offset = 0 len = 16)
            %c1_ul2 = arith.constant 1 : i32
            aie.use_lock(%test_cons_lock, Release, %c1_ul2)
            aie.next_bd ^bb1
         ^bb2:
            aie.end
      }
   }
}
