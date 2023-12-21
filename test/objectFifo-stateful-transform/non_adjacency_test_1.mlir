//===- non_adjacency_test_1.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: February 10th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @objfifo_cons : memref<16xi32>
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_16:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_15]] {
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_6]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_7]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_22:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] {
// CHECK:               aie.use_lock(%[[VAL_4]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_4]], Release, 0)
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, 1)
// CHECK:               func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_5]], Release, 0)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @non_adjacency {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @objfifo (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @objfifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo (Produce, 1)
            }

            aie.end
        }

        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = aie.objectfifo.acquire @objfifo (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo (Consume, 1)
            }

            aie.end
        }
    }
}
