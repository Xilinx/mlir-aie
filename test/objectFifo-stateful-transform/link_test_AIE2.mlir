//===- link_test_AIE2.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 31st 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @link_AIE2 {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(0, 0)
// CHECK:     %1 = AIE.tile(0, 1)
// CHECK:     %2 = AIE.tile(0, 2)
// CHECK:     %3 = AIE.tile(0, 3)
// CHECK:     memref.global "public" @mem_in : memref<3000xi32>
// CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
// CHECK:     AIE.flow(%0, DMA : 0, %2, DMA : 0)
// CHECK:     %4 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "mem_in_prod_lock"}
// CHECK:     %5 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "mem_in_cons_lock"}
// CHECK:     memref.global "public" @mem_in_1_cons : memref<3000xi32>
// CHECK:     %6 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_0"} : memref<3000xi32>
// CHECK:     %7 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_1"} : memref<3000xi32>
// CHECK:     %8 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_2"} : memref<3000xi32>
// CHECK:     %9 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_3"} : memref<3000xi32>
// CHECK:     %10 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_4"} : memref<3000xi32>
// CHECK:     %11 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_5"} : memref<3000xi32>
// CHECK:     %12 = AIE.buffer(%1) {sym_name = "mem_in_1_cons_buff_6"} : memref<3000xi32>
// CHECK:     %13 = AIE.lock(%1, 0) {init = 7 : i32, sym_name = "mem_in_1_cons_prod_lock"}
// CHECK:     %14 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "mem_in_1_cons_cons_lock"}
// CHECK:     memref.global "public" @mem_in_0_cons : memref<3000xi32>
// CHECK:     %15 = AIE.buffer(%2) {sym_name = "mem_in_0_cons_buff_0"} : memref<3000xi32>
// CHECK:     %16 = AIE.buffer(%2) {sym_name = "mem_in_0_cons_buff_1"} : memref<3000xi32>
// CHECK:     %17 = AIE.lock(%2, 0) {init = 2 : i32, sym_name = "mem_in_0_cons_prod_lock"}
// CHECK:     %18 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "mem_in_0_cons_cons_lock"}
// CHECK:     memref.global "public" @mem_out : memref<3000xi32>
// CHECK:     AIE.flow(%1, DMA : 0, %3, DMA : 0)
// CHECK:     memref.global "public" @mem_out_cons : memref<3000xi32>
// CHECK:     %19 = AIE.buffer(%3) {sym_name = "mem_out_cons_buff_0"} : memref<3000xi32>
// CHECK:     %20 = AIE.buffer(%3) {sym_name = "mem_out_cons_buff_1"} : memref<3000xi32>
// CHECK:     %21 = AIE.buffer(%3) {sym_name = "mem_out_cons_buff_2"} : memref<3000xi32>
// CHECK:     %22 = AIE.buffer(%3) {sym_name = "mem_out_cons_buff_3"} : memref<3000xi32>
// CHECK:     %23 = AIE.lock(%3, 0) {init = 4 : i32, sym_name = "mem_out_cons_prod_lock"}
// CHECK:     %24 = AIE.lock(%3, 1) {init = 0 : i32, sym_name = "mem_out_cons_cons_lock"}
// CHECK:     %25 = AIE.core(%2) {
// CHECK:       %c11_i32 = arith.constant 11 : i32
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       AIE.useLock(%18, AcquireGreaterEqual, 1)
// CHECK:       memref.store %c11_i32, %15[%c0] : memref<3000xi32>
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     AIE.shimDMAAllocation(@mem_in, MM2S, 0, 0)
// CHECK:     %26 = AIE.core(%3) {
// CHECK:       %c11_i32 = arith.constant 11 : i32
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 3)
// CHECK:       memref.store %c11_i32, %19[%c0] : memref<3000xi32>
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %27 = AIE.mem(%2) {
// CHECK:       %30 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%17, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%15 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%18, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%17, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%16 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%18, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %28 = AIE.memTileDMA(%1) {
// CHECK:       %30 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb8)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb7
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%6 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%8 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%9 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%10 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%11 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%12 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%14, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb8:  // pred: ^bb0
// CHECK:       %31 = AIE.dmaStart(MM2S, 0, ^bb9, ^bb16)
// CHECK:     ^bb9:  // 2 preds: ^bb8, ^bb15
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%6 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb10
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%8 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb12
// CHECK:     ^bb12:  // pred: ^bb11
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%9 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb13
// CHECK:     ^bb13:  // pred: ^bb12
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%10 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb14
// CHECK:     ^bb14:  // pred: ^bb13
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%11 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb15
// CHECK:     ^bb15:  // pred: ^bb14
// CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%12 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb9
// CHECK:     ^bb16:  // pred: ^bb8
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %29 = AIE.mem(%3) {
// CHECK:       %30 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       AIE.useLock(%23, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%19 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%24, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%23, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%20 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%24, Release, 1)
// CHECK:       AIE.nextBd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       AIE.useLock(%23, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%21 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%24, Release, 1)
// CHECK:       AIE.nextBd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       AIE.useLock(%23, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%22 : memref<3000xi32>, 0, 3000>, 0)
// CHECK:       AIE.useLock(%24, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }              

module @link_AIE2 {
    AIE.device(xcve2302) {
        %tile00 = AIE.tile(0, 0)
        %tile01 = AIE.tile(0, 1)
        %tile02 = AIE.tile(0, 2)
        %tile03 = AIE.tile(0, 3)

        AIE.objectFifo @mem_in (%tile00, {%tile02, %tile01}, [2,2,7]) : !AIE.objectFifo<memref<3000xi32>>
        AIE.objectFifo @mem_out (%tile01, {%tile03}, 7 : i32) : !AIE.objectFifo<memref<3000xi32>>
        AIE.objectFifo.link({%objFifo}, {%objFifo2}) : ({!AIE.objectFifo<memref<3000xi32>>}, {!AIE.objectFifo<memref<3000xi32>>})

        %core02 = AIE.core(%tile02) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index

            %subview = AIE.objectFifo.acquire @mem_in (Consume, 1) : !AIE.objectFifoSubview<memref<3000xi32>>
            %subview_obj = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<3000xi32>> -> memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            AIE.end
        }

        %core03 = AIE.core(%tile03) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index

            %subview = AIE.objectFifo.acquire @mem_out (Consume, 3) : !AIE.objectFifoSubview<memref<3000xi32>>
            %subview_obj = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<3000xi32>> -> memref<3000xi32>
            memref.store %v11, %subview_obj[%c0] : memref<3000xi32>
            AIE.end
        }
    }
}
