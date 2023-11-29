//===- tileDMA_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: September 22nd 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK:           %[[VAL_6:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_0]], 4) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_12:.*]] = AIE.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = AIE.lock(%[[VAL_0]], 1)
// CHECK:           %[[VAL_14:.*]] = AIE.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_0]], 2)
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_16:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_22:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] {
// CHECK:               AIE.useLock(%[[VAL_8]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_6]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:               AIE.useLock(%[[VAL_9]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_7]]) : (memref<16xi32>) -> ()
// CHECK:               AIE.useLock(%[[VAL_9]], Release, 1)
// CHECK:             }
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_24:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_11]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_10]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 0)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_13]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_12]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_13]], Release, 0)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_25:.*]] = AIE.dmaStart(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:             AIE.useLock(%[[VAL_15]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_14]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb5:  // pred: ^bb3
// CHECK:             %[[VAL_26:.*]] = AIE.dmaStart(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:             AIE.useLock(%[[VAL_8]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 0)
// CHECK:             AIE.nextBd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:             AIE.useLock(%[[VAL_9]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_7]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_9]], Release, 0)
// CHECK:             AIE.nextBd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = AIE.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_28:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_4]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_2]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_3]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @tileDMA_channels {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile33 = AIE.tile(3, 3)

        %buff0 = AIE.buffer(%tile12) : memref<16xi32>
        %lock0 = AIE.lock(%tile12, 0)
        %buff1 = AIE.buffer(%tile12) : memref<16xi32>
        %lock1 = AIE.lock(%tile12, 1)
        %buff2 = AIE.buffer(%tile12) : memref<16xi32>
        %lock2 = AIE.lock(%tile12, 2)

        AIE.objectfifo @objfifo (%tile12, {%tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

        func.func @some_work(%lineOut : memref<16xi32>) -> () {
            return
        }

        %core12 = AIE.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %subview = AIE.objectfifo.acquire @objfifo (Produce, 1) : !AIE.objectfifosubview<memref<16xi32>>
                %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                AIE.objectfifo.release @objfifo (Produce, 1)
            }
            
            AIE.end
        }

        %mem12 = AIE.mem(%tile12) {
            %dma1 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
        ^bb1:
            AIE.useLock(%lock0, Acquire, 1)
            AIE.dmaBd(<%buff0 : memref<16xi32>, 0, 16>, 0)
            AIE.useLock(%lock0, Release, 0)
            AIE.nextBd ^bb2
        ^bb2:
            AIE.useLock(%lock1, Acquire, 1)
            AIE.dmaBd(<%buff1 : memref<16xi32>, 0, 16>, 0)
            AIE.useLock(%lock1, Release, 0)
            AIE.nextBd ^bb1
        ^bb3:
            %dma2 = AIE.dmaStart(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            AIE.useLock(%lock2, Acquire, 0)
            AIE.dmaBd(<%buff2 : memref<16xi32>, 0, 16>, 0)
            AIE.useLock(%lock2, Release, 1)
            AIE.nextBd ^bb4
        ^bb5:
            AIE.end
        }
    }
}
