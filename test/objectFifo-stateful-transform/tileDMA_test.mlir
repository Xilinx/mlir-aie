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

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_0]], 4) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_0]], 1)
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_0]]) : memref<16xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_0]], 2)
// CHECK:           aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_1]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_16:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_20:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_22:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] {
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_6]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, 0)
// CHECK:               func.call @some_work(%[[VAL_7]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_9]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_13]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:             aie.use_lock(%[[VAL_15]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:  // pred: ^bb3
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_9]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @tileDMA_channels {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)

        %buff0 = aie.buffer(%tile12) : memref<16xi32>
        %lock0 = aie.lock(%tile12, 0)
        %buff1 = aie.buffer(%tile12) : memref<16xi32>
        %lock1 = aie.lock(%tile12, 1)
        %buff2 = aie.buffer(%tile12) : memref<16xi32>
        %lock2 = aie.lock(%tile12, 2)

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

        %mem12 = aie.mem(%tile12) {
            %dma1 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
        ^bb1:
            aie.use_lock(%lock0, Acquire, 1)
            aie.dma_bd(%buff0 : memref<16xi32>) { len = 16 : i32 }
            aie.use_lock(%lock0, Release, 0)
            aie.next_bd ^bb2
        ^bb2:
            aie.use_lock(%lock1, Acquire, 1)
            aie.dma_bd(%buff1 : memref<16xi32>) { len = 16 : i32 }
            aie.use_lock(%lock1, Release, 0)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            aie.use_lock(%lock2, Acquire, 0)
            aie.dma_bd(%buff2 : memref<16xi32>) { len = 16 : i32 }
            aie.use_lock(%lock2, Release, 1)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
