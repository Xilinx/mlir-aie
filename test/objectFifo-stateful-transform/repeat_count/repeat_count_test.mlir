//===- repeat_count_test.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.lock(%[[VAL_0]]) {init = 3 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_4]]) {init = 1 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]]) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_8:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_10:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 3 : i32
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_12]] to %[[VAL_10]] step %[[VAL_11]] {
// CHECK:               aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:               func.call @some_work(%[[VAL_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_3]], Release, %[[VAL_13]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_2]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[VAL_19]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @repeatCount {
 aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
       return
    }

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %height = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %height step %c1 {
         %elem0 = aie.objectfifo.acquire @of1(Produce) : memref<16xi32>
         func.call @some_work(%elem0) : (memref<16xi32>) -> ()
         aie.objectfifo.release @of1(Produce) [1]
      }

      aie.end
   }
 }
}
