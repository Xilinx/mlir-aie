//===- broadcast_enforced_depths.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: September 5th 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK-DAG:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK-DAG:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK-DAG:           %[[VAL_2:.*]] = aie.tile(1, 4)
// CHECK-DAG:           %[[VAL_3:.*]] = aie.tile(3, 2)
// CHECK-DAG:           %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK-DAG:           aie.buffer @[[VAL_5:broadcast_of_0_cons_buff_0]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_6:broadcast_of_0_cons_buff_1]](%[[VAL_0]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_7:broadcast_of_0_cons_lock_0]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_8:broadcast_of_0_cons_lock_1]](%[[VAL_0]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_9:broadcast_of_1_cons_buff_0]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_10:broadcast_of_1_cons_buff_1]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_11:broadcast_of_1_cons_buff_2]](%[[VAL_2]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_12:broadcast_of_1_cons_lock_0]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_13:broadcast_of_1_cons_lock_1]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_14:broadcast_of_1_cons_lock_2]](%[[VAL_2]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_15:broadcast_of_2_cons_buff_0]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_16:broadcast_of_2_cons_buff_1]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_17:broadcast_of_2_cons_buff_2]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_18:broadcast_of_2_cons_buff_3]](%[[VAL_3]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_19:broadcast_of_2_cons_lock_0]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_20:broadcast_of_2_cons_lock_1]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_21:broadcast_of_2_cons_lock_2]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_22:broadcast_of_2_cons_lock_3]](%[[VAL_3]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_23:broadcast_of_3_cons_buff_0]](%[[VAL_4]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_24:broadcast_of_3_cons_buff_1]](%[[VAL_4]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_25:broadcast_of_3_cons_buff_2]](%[[VAL_4]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_26:broadcast_of_3_cons_lock_0]](%[[VAL_4]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_27:broadcast_of_3_cons_lock_1]](%[[VAL_4]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_28:broadcast_of_3_cons_lock_2]](%[[VAL_4]]) {init = 0 : i32}
// CHECK-DAG:           aie.buffer @[[VAL_29:broadcast_of_buff_0]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.buffer @[[VAL_30:broadcast_of_buff_1]](%[[VAL_1]]) : memref<16xi32>
// CHECK-DAG:           aie.lock @[[VAL_31:broadcast_of_lock_0]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.lock @[[VAL_32:broadcast_of_lock_1]](%[[VAL_1]]) {init = 0 : i32}
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK-DAG:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           func.func @some_work(%[[VAL_33:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.core(%[[VAL_1]]) {
// CHECK-DAG:             %[[VAL_35:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_37:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_38:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_39:.*]] = %[[VAL_35]] to %[[VAL_37]] step %[[VAL_38]] {
// CHECK:               aie.use_lock(@[[VAL_31]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_31]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_32]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_32]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[VAL_41:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_43:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_44:.*]] = arith.constant 2 : index
// CHECK:             scf.for %[[VAL_45:.*]] = %[[VAL_41]] to %[[VAL_43]] step %[[VAL_44]] {
// CHECK:               aie.use_lock(@[[VAL_7]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_8]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_8]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = aie.core(%[[VAL_2]]) {
// CHECK-DAG:             %[[VAL_47:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_49:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_50:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[VAL_51:.*]] = %[[VAL_47]] to %[[VAL_49]] step %[[VAL_50]] {
// CHECK:               aie.use_lock(@[[VAL_12]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_13]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_14]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_12]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_14]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_13]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_14]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_14]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = aie.core(%[[VAL_3]]) {
// CHECK-DAG:             %[[VAL_53:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_55:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_56:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_57:.*]] = %[[VAL_53]] to %[[VAL_55]] step %[[VAL_56]] {
// CHECK:               aie.use_lock(@[[VAL_19]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_20]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_21]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_19]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_22]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_20]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_19]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_21]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_20]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_22]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = aie.core(%[[VAL_4]]) {
// CHECK-DAG:             %[[VAL_59:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[VAL_61:.*]] = arith.constant 12 : index
// CHECK-DAG:             %[[VAL_62:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[VAL_63:.*]] = %[[VAL_59]] to %[[VAL_61]] step %[[VAL_62]] {
// CHECK:               aie.use_lock(@[[VAL_26]], Acquire, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_27]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_26]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_28]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_27]], Release, %{{.*}})
// CHECK:               aie.use_lock(@[[VAL_26]], Acquire, %{{.*}})
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               func.call @some_work(%{{[^ ,)]+}}) : (memref<16xi32>) -> ()
// CHECK:               aie.use_lock(@[[VAL_28]], Release, %{{.*}})
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_65:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_31]], Acquire, %{{.*}})
// CHECK:             %[[VAL_29_ACCESS_0:.*]] = aie.get_buffer @[[VAL_29]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_29_ACCESS_0]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_31]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_32]], Acquire, %{{.*}})
// CHECK:             %[[VAL_30_ACCESS_1:.*]] = aie.get_buffer @[[VAL_30]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_30_ACCESS_1]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_32]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_66:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_67:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(@[[VAL_7]], Acquire, %{{.*}})
// CHECK:             %[[VAL_5_ACCESS_2:.*]] = aie.get_buffer @[[VAL_5]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_5_ACCESS_2]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_7]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_8]], Acquire, %{{.*}})
// CHECK:             %[[VAL_6_ACCESS_3:.*]] = aie.get_buffer @[[VAL_6]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_6_ACCESS_3]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_8]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_68:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_69:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(@[[VAL_12]], Acquire, %{{.*}})
// CHECK:             %[[VAL_9_ACCESS_4:.*]] = aie.get_buffer @[[VAL_9]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_9_ACCESS_4]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_12]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_13]], Acquire, %{{.*}})
// CHECK:             %[[VAL_10_ACCESS_5:.*]] = aie.get_buffer @[[VAL_10]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_10_ACCESS_5]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_13]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(@[[VAL_14]], Acquire, %{{.*}})
// CHECK:             %[[VAL_11_ACCESS_6:.*]] = aie.get_buffer @[[VAL_11]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_11_ACCESS_6]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_14]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_70:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_71:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:             aie.use_lock(@[[VAL_19]], Acquire, %{{.*}})
// CHECK:             %[[VAL_15_ACCESS_7:.*]] = aie.get_buffer @[[VAL_15]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_15_ACCESS_7]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_19]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_20]], Acquire, %{{.*}})
// CHECK:             %[[VAL_16_ACCESS_8:.*]] = aie.get_buffer @[[VAL_16]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_16_ACCESS_8]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_20]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(@[[VAL_21]], Acquire, %{{.*}})
// CHECK:             %[[VAL_17_ACCESS_9:.*]] = aie.get_buffer @[[VAL_17]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_17_ACCESS_9]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_21]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:             aie.use_lock(@[[VAL_22]], Acquire, %{{.*}})
// CHECK:             %[[VAL_18_ACCESS_10:.*]] = aie.get_buffer @[[VAL_18]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_18_ACCESS_10]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_22]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_73:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(@[[VAL_26]], Acquire, %{{.*}})
// CHECK:             %[[VAL_23_ACCESS_11:.*]] = aie.get_buffer @[[VAL_23]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_23_ACCESS_11]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_26]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(@[[VAL_27]], Acquire, %{{.*}})
// CHECK:             %[[VAL_24_ACCESS_12:.*]] = aie.get_buffer @[[VAL_24]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_24_ACCESS_12]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_27]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(@[[VAL_28]], Acquire, %{{.*}})
// CHECK:             %[[VAL_25_ACCESS_13:.*]] = aie.get_buffer @[[VAL_25]] : memref<16xi32>
// CHECK:             aie.dma_bd(%[[VAL_25_ACCESS_13]] : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:             aie.use_lock(@[[VAL_28]], Release, %{{.*}})
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @broadcast {
 aie.device(xcvc1902) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile14 = aie.tile(1, 4)
    %tile32 = aie.tile(3, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @broadcast_of (%tile13, {%tile12, %tile14, %tile32, %tile33}, [2, 2, 3, 4, 3]) : !aie.objectfifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core13 = aie.core(%tile13) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %elem0 = aie.objectfifo.acquire @broadcast_of (Produce, 1) : memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Produce, 1)
        }

        aie.end
    }

    %core12 = aie.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %elem0 = aie.objectfifo.acquire @broadcast_of (Consume, 1) : memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }

        aie.end
    }

    %core14 = aie.core(%tile14) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %elem0, %elem1 = aie.objectfifo.acquire @broadcast_of (Consume, 2) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 2)
        }

        aie.end
    }

    %core32 = aie.core(%tile32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %elem0, %elem1, %elem2 = aie.objectfifo.acquire @broadcast_of (Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            func.call @some_work(%elem2) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }

        aie.end
    }

    %core33 = aie.core(%tile33) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %elem0, %elem1 = aie.objectfifo.acquire @broadcast_of (Consume, 2) : memref<16xi32>, memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            func.call @some_work(%elem1) : (memref<16xi32>) -> ()
            aie.objectfifo.release @broadcast_of (Consume, 1)
        }

        aie.end
    }
 }
}
