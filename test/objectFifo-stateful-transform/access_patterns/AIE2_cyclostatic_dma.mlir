//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In this test, data is exchanged the same as in AIE2_cyclostatic_l1, but
// tiles are farther apart and have to use the network/DMAs to communicate.

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s
// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo_cons_buff_0"} : memref<i32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo_cons_buff_1"} : memref<i32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo_cons_buff_2"} : memref<i32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_5]]) {init = 3 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "buf83"} : memref<4xi32>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_14:.*]] = arith.constant 88 : i32
// CHECK:             %[[VAL_15:.*]] = arith.constant 77 : i32
// CHECK:             %[[VAL_16:.*]] = arith.constant 66 : i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 55 : i32
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             memref.store %[[VAL_17]], %[[VAL_1]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             memref.store %[[VAL_16]], %[[VAL_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             memref.store %[[VAL_15]], %[[VAL_1]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_13]])
// CHECK:             memref.store %[[VAL_14]], %[[VAL_2]][] : memref<i32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_13]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.core(%[[VAL_5]]) {
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_20:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_21:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_24:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK:             memref.store %[[VAL_25]], %[[VAL_11]]{{\[}}%[[VAL_23]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_19]])
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_24]])
// CHECK:             %[[VAL_26:.*]] = memref.load %[[VAL_7]][] : memref<i32>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK:             memref.store %[[VAL_26]], %[[VAL_11]]{{\[}}%[[VAL_22]]] : memref<4xi32>
// CHECK:             memref.store %[[VAL_27]], %[[VAL_11]]{{\[}}%[[VAL_21]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_24]])
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             %[[VAL_28:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK:             memref.store %[[VAL_28]], %[[VAL_11]]{{\[}}%[[VAL_20]]] : memref<4xi32>
// CHECK:             aie.use_lock(%[[VAL_9]], Release, %[[VAL_19]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_30:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<i32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_30]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<i32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_30]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<i32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<i32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<i32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_33]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie2_cyclostatic_dma {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile83 = aie.tile(8, 3)  // consumer tile
        %buf83  = aie.buffer(%tile83) {sym_name = "buf83"} : memref<4xi32>

        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile22, {%tile83}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer core
        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32

            // Push 55
            %subview0_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            // Push 66
            %subview1_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            // Push 77
            %subview2_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            // Push 88
            %subview3_obj = aie.objectfifo.acquire @fifo(Produce) : memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            aie.objectfifo.release @fifo(Produce) [1]

            aie.end
        }

        // Consumer core
        %core28 = aie.core(%tile83) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Pop 1 object off queue
            %subview0_obj = aie.objectfifo.acquire @fifo(Consume) : memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v55, %buf83[%i0] : memref<4xi32>
            aie.objectfifo.release @fifo(Consume) [1]

            // Pop 2 objects off queue
            %subview1_obj0, %subview1_obj1 = aie.objectfifo.acquire @fifo(Consume) : memref<i32>, memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf83[%i1] : memref<4xi32>
            memref.store %v77, %buf83[%i2] : memref<4xi32>
            aie.objectfifo.release @fifo(Consume) [2]

            // Pop 1 object off queue
            %subview2_obj = aie.objectfifo.acquire @fifo(Consume) : memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf83[%i3] : memref<4xi32>
            aie.objectfifo.release @fifo(Consume) [1]

            aie.end
        }
    }
}
