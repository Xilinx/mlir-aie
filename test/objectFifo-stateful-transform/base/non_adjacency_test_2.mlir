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
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(3, 3)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_2"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "objfifo_cons_buff_3"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_0"}
// CHECK-DAG:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_1"}
// CHECK-DAG:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_2"}
// CHECK-DAG:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "objfifo_cons_lock_3"}
// CHECK-DAG:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_0"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "objfifo_buff_1"} : memref<16xi32>
// CHECK-DAG:           %[[VAL_12:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_0"}
// CHECK-DAG:           %[[VAL_13:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "objfifo_lock_1"}
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_14:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_18:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_19:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] {
// CHECK:               aie.use_lock(%[[VAL_12]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%[[VAL_10]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:               aie.use_lock(%[[VAL_13]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%[[VAL_11]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_13]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.core(%[[VAL_1]]) {
// CHECK-DAG:             %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_24:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_25:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_26:.*]] = %[[VAL_22]] to %[[VAL_24]] step %[[VAL_25]] {
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(%[[VAL_8]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%[[VAL_2]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %{{.*}})
// CHECK:               aie.use_lock(%[[VAL_9]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%[[VAL_3]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %{{.*}})
// CHECK:               aie.use_lock(%[[VAL_6]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%[[VAL_4]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_8]], Release, %{{.*}})
// CHECK:               aie.use_lock(%[[VAL_7]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%[[VAL_5]]) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(%[[VAL_9]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_12]], Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[VAL_12]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_13]], Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_11]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[VAL_13]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[VAL_8]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:             aie.use_lock(%[[VAL_9]], Acquire, %{{.*}})
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
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
                %elem0 = aie.objectfifo.acquire @objfifo (Produce, 1) : memref<16xi32>
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
                %elem0, %elem1, %elem2 = aie.objectfifo.acquire @objfifo (Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
                func.call @some_work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @objfifo (Consume, 1)
            }

            aie.end
        }
    }
}
