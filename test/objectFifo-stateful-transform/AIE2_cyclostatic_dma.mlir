//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// In this test, data is exchanged the same as in AIE2_cyclostatic_l1, but
// tiles are farther apart and have to use the network/DMAs to communicate.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// CHECK: module @aie2_cyclostatic_dma {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %[[t0:.*]] = AIE.tile(2, 2)
// CHECK:     %[[t1:.*]] = AIE.tile(8, 3)
// CHECK:     %[[buf1_0:.*]] = AIE.buffer(%[[t1]]) {sym_name = "fifo_cons_buff_0"} : memref<i32>
// CHECK:     %[[buf1_1:.*]] = AIE.buffer(%[[t1]]) {sym_name = "fifo_cons_buff_1"} : memref<i32>
// CHECK:     %[[buf1_2:.*]] = AIE.buffer(%[[t1]]) {sym_name = "fifo_cons_buff_2"} : memref<i32>
// CHECK:     %[[C_PL:.*]] = AIE.lock(%[[t1]], 0) {init = 3 : i32, sym_name = "fifo_cons_prod_lock"}
// CHECK:     %[[C_CL:.*]] = AIE.lock(%[[t1]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock"}
// CHECK:     %[[buf0_0:.*]] = AIE.buffer(%[[t0]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:     %[[buf0_1:.*]] = AIE.buffer(%[[t0]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:     %[[PL:.*]] = AIE.lock(%[[t0]], 0) {init = 2 : i32, sym_name = "fifo_prod_lock"}
// CHECK:     %[[CL:.*]] = AIE.lock(%[[t0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:     AIE.flow(%[[t0]], DMA : 0, %[[t1]], DMA : 0)
// CHECK:     %[[c0:.*]] = AIE.core(%[[t0]]) {
// CHECK:       AIE.useLock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock(%[[CL]], Release, 1)
// CHECK:       AIE.useLock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock(%[[CL]], Release, 1)
// CHECK:       AIE.useLock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock(%[[CL]], Release, 1)
// CHECK:       AIE.useLock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock(%[[CL]], Release, 1)
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %[[c1:.*]] = AIE.core(%[[t1]]) {
// CHECK:       AIE.useLock(%[[C_CL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock(%[[C_PL]], Release, 1)
// CHECK:       AIE.useLock(%[[C_CL]], AcquireGreaterEqual, 2)
// CHECK:       AIE.useLock(%[[C_PL]], Release, 2)
// CHECK:       AIE.useLock(%[[C_CL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.useLock(%[[C_PL]], Release, 1)
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %[[m0:.*]] = AIE.mem(%[[t0]]) {
// CHECK:       %[[dma0:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%[[CL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[buf0_0]] : memref<i32>, 0, 1>, 0)
// CHECK:       AIE.useLock(%[[PL]], Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[CL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[buf0_1:.*]] : memref<i32>, 0, 1>, 0)
// CHECK:       AIE.useLock(%[[PL]], Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %[[m1:.*]] = AIE.mem(%[[t1]]) {
// CHECK:       %[[dma1:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       AIE.useLock(%[[C_PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[buf1_0:.*]] : memref<i32>, 0, 1>, 0)
// CHECK:       AIE.useLock(%[[C_CL]], Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[C_PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[buf1_1]] : memref<i32>, 0, 1>, 0)
// CHECK:       AIE.useLock(%[[C_CL]], Release, 1)
// CHECK:       AIE.nextBd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       AIE.useLock(%[[C_PL]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%[[buf1_2]] : memref<i32>, 0, 1>, 0)
// CHECK:       AIE.useLock(%[[C_CL]], Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @aie2_cyclostatic_dma {
    AIE.device(xcve2302) {

        %tile22 = AIE.tile(2, 2)  // producer tile
        %tile83 = AIE.tile(8, 3)  // consumer tile
        %buf83  = AIE.buffer(%tile83) {sym_name = "buf83"} : memref<4xi32>

        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        AIE.objectfifo @fifo (%tile22, {%tile83}, 4 : i32) : !AIE.objectfifo<memref<i32>>

        // Producer core
        %core22 = AIE.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            
            // Push 55
            %subview0 = AIE.objectfifo.acquire @fifo (Produce, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview0_obj = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            AIE.objectfifo.release @fifo (Produce, 1)

            // Push 66
            %subview1 = AIE.objectfifo.acquire @fifo (Produce, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview1_obj = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            AIE.objectfifo.release @fifo (Produce, 1)

            // Push 77
            %subview2 = AIE.objectfifo.acquire @fifo (Produce, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview2_obj = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            AIE.objectfifo.release @fifo (Produce, 1)

            // Push 88
            %subview3 = AIE.objectfifo.acquire @fifo (Produce, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview3_obj = AIE.objectfifo.subview.access %subview3[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            AIE.objectfifo.release @fifo (Produce, 1)

            AIE.end
        }

        // Consumer core
        %core28 = AIE.core(%tile83) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Pop 1 object off queue
            %subview0 = AIE.objectfifo.acquire @fifo (Consume, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview0_obj = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v55, %buf83[%i0] : memref<4xi32>
            AIE.objectfifo.release @fifo (Consume, 1)

            // Pop 2 objects off queue
            %subview1 = AIE.objectfifo.acquire @fifo (Consume, 2) : !AIE.objectfifosubview<memref<i32>>
            %subview1_obj0 = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %subview1_obj1 = AIE.objectfifo.subview.access %subview1[1] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf83[%i1] : memref<4xi32>
            memref.store %v77, %buf83[%i2] : memref<4xi32>
            AIE.objectfifo.release @fifo (Consume, 2)

            // Pop 1 object off queue
            %subview2 = AIE.objectfifo.acquire @fifo (Consume, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview2_obj = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf83[%i3] : memref<4xi32>
            AIE.objectfifo.release @fifo (Consume, 1)

            AIE.end
        }
    }
}
