//===- tileDMA_test.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 22nd 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(3, 3)
// CHECK-DAG:           aie.buffer @[[VAL_2:objfifo_cons_buff_0]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_3:objfifo_cons_buff_1]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_4:objfifo_cons_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_5:objfifo_cons_lock_1]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_6:objfifo_buff_0]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_7:objfifo_buff_1]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_8:objfifo_lock_0]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_9:objfifo_lock_1]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_10:.*]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_11:.*]](%[[VAL_0]], 0)
// CHECK-DAG:           aie.buffer @[[VAL_12:.*]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_13:.*]](%[[VAL_0]], 1)
// CHECK-DAG:           aie.buffer @[[VAL_14:.*]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_15:.*]](%[[VAL_0]], 2)
// CHECK-DAG:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_16:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_20:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_21:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_22:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] {
// CHECK:               aie.use_lock(@[[VAL_8]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_8]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_9]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_9]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_11]], Acquire, %{{.*}})
// CHECK:             %[[VAL_10_ACCESS_0:.*]] = aie.get_buffer @[[VAL_10]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_10_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_11]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_13]], Acquire, %{{.*}})
// CHECK:             %[[VAL_12_ACCESS_1:.*]] = aie.get_buffer @[[VAL_12]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_12_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:             aie.use_lock(@[[VAL_15]], Acquire, %{{.*}})
// CHECK:             %[[VAL_14_ACCESS_2:.*]] = aie.get_buffer @[[VAL_14]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_14_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_15]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:  // pred: ^bb3
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:             aie.use_lock(@[[VAL_8]], Acquire, %{{.*}})
// CHECK:             %[[VAL_6_ACCESS_3:.*]] = aie.get_buffer @[[VAL_6]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_6_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_8]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:             aie.use_lock(@[[VAL_9]], Acquire, %{{.*}})
// CHECK:             %[[VAL_7_ACCESS_4:.*]] = aie.get_buffer @[[VAL_7]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_7_ACCESS_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_9]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_4]], Acquire, %{{.*}})
// CHECK:             %[[VAL_2_ACCESS_5:.*]] = aie.get_buffer @[[VAL_2]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_2_ACCESS_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_4]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_5]], Acquire, %{{.*}})
// CHECK:             %[[VAL_3_ACCESS_6:.*]] = aie.get_buffer @[[VAL_3]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_3_ACCESS_6]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_5]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @tileDMA_channels {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)

        aie.buffer @buff0(%tile12) : memref<16xi32>
        aie.lock @lock0(%tile12, 0)
        aie.buffer @buff1(%tile12) : memref<16xi32>
        aie.lock @lock1(%tile12, 1)
        aie.buffer @buff2(%tile12) : memref<16xi32>
        aie.lock @lock2(%tile12, 2)

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

        %mem12 = aie.mem(%tile12) {
            %dma1 = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
        ^bb1:
            %c1_ul1 = arith.constant 1 : i32
            aie.use_lock(@lock0, Acquire, %c1_ul1)
            %buff0 = aie.get_buffer @buff0 : memref<16xi32>
            aie.dma_bd(%buff0 : memref<16xi32> offset = 0 len = 16)
            %c0_ul2 = arith.constant 0 : i32
            aie.use_lock(@lock0, Release, %c0_ul2)
            aie.next_bd ^bb2
        ^bb2:
            %c1_ul3 = arith.constant 1 : i32
            aie.use_lock(@lock1, Acquire, %c1_ul3)
            %buff1 = aie.get_buffer @buff1 : memref<16xi32>
            aie.dma_bd(%buff1 : memref<16xi32> offset = 0 len = 16)
            %c0_ul4 = arith.constant 0 : i32
            aie.use_lock(@lock1, Release, %c0_ul4)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            %c0_ul5 = arith.constant 0 : i32
            aie.use_lock(@lock2, Acquire, %c0_ul5)
            %buff2 = aie.get_buffer @buff2 : memref<16xi32>
            aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
            %c1_ul6 = arith.constant 1 : i32
            aie.use_lock(@lock2, Release, %c1_ul6)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
