//===- non_adjacency_test_2.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: May 24th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "objfifo_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "objfifo_cons_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_2"}
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_3"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_14:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_17:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[VAL_21:.*]] = %[[VAL_18]] to %[[VAL_17]] step %[[VAL_16]] {
// CHECK:               aie.use_lock(%[[VAL_3]], Acquire, %[[VAL_19]])
// CHECK:               func.call @some_work(%[[VAL_1]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_3]], Release, %[[VAL_20]])
// CHECK:               aie.use_lock(%[[VAL_4]], Acquire, %[[VAL_19]])
// CHECK:               func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_4]], Release, %[[VAL_20]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.core(%[[VAL_5]]) {
// CHECK:             %[[VAL_23:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_25:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[VAL_28:.*]] = %[[VAL_25]] to %[[VAL_24]] step %[[VAL_23]] {
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_27]])
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_27]])
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, %[[VAL_27]])
// CHECK:               func.call @some_work(%[[VAL_6]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_10]], Release, %[[VAL_26]])
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, %[[VAL_27]])
// CHECK:               func.call @some_work(%[[VAL_7]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_11]], Release, %[[VAL_26]])
// CHECK:               aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_27]])
// CHECK:               func.call @some_work(%[[VAL_8]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, %[[VAL_26]])
// CHECK:               aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_27]])
// CHECK:               func.call @some_work(%[[VAL_9]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %[[VAL_26]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_3]], Acquire, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_31]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], Acquire, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_31]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_34:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], Acquire, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_11]], Acquire, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, %[[VAL_34]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32> offset = 0 len = 16)
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %[[VAL_35]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
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
                %elem0 = aie.objectfifo.acquire @objfifo(Produce) : memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo(Produce) [1]
            }

            aie.end
        }

        %core33 = aie.core(%tile33) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %height = arith.constant 12 : index

            scf.for %indexInHeight = %c0 to %height step %c1 {
                %elem0, %elem1, %elem2 = aie.objectfifo.acquire @objfifo(Consume) : memref<16xi32>, memref<16xi32>, memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo(Consume) [1]
            }

            aie.end
        }
    }
}
