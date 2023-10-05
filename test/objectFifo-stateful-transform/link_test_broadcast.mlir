//===- link_test_broadcast.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: module @link_broadcast {
//CHECK:   AIE.device(xcve2302) {
//CHECK:     memref.global "public" @skip_connection_cons : memref<16xi32>
//CHECK:     memref.global "public" @skip_connection : memref<16xi32>
//CHECK:     memref.global "public" @link2_0_cons : memref<16xi32>
//CHECK:     memref.global "public" @link2_1_cons : memref<16xi32>
//CHECK:     memref.global "public" @link2 : memref<16xi32>
//CHECK:     memref.global "public" @link1_cons : memref<48xi32>
//CHECK:     memref.global "public" @link1 : memref<48xi32>
//CHECK:     %0 = AIE.tile(2, 0)
//CHECK:     %1 = AIE.tile(2, 1)
//CHECK:     %2 = AIE.tile(2, 2)
//CHECK:     %3 = AIE.tile(3, 3)
//CHECK:     %4 = AIE.buffer(%3) {sym_name = "skip_connection_cons_buff_0"} : memref<16xi32>
//CHECK:     %5 = AIE.buffer(%3) {sym_name = "skip_connection_cons_buff_1"} : memref<16xi32>
//CHECK:     %6 = AIE.lock(%3, 2) {init = 2 : i32, sym_name = "skip_connection_cons_prod_lock"}
//CHECK:     %7 = AIE.lock(%3, 3) {init = 0 : i32, sym_name = "skip_connection_cons_cons_lock"}
//CHECK:     %8 = AIE.buffer(%2) {sym_name = "skip_connection_buff_0"} : memref<16xi32>
//CHECK:     %9 = AIE.buffer(%2) {sym_name = "skip_connection_buff_1"} : memref<16xi32>
//CHECK:     %10 = AIE.lock(%2, 2) {init = 2 : i32, sym_name = "skip_connection_prod_lock"}
//CHECK:     %11 = AIE.lock(%2, 3) {init = 0 : i32, sym_name = "skip_connection_cons_lock"}
//CHECK:     %12 = AIE.buffer(%2) {sym_name = "link2_0_cons_buff_0"} : memref<16xi32>
//CHECK:     %13 = AIE.buffer(%2) {sym_name = "link2_0_cons_buff_1"} : memref<16xi32>
//CHECK:     %14 = AIE.lock(%2, 0) {init = 2 : i32, sym_name = "link2_0_cons_prod_lock"}
//CHECK:     %15 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "link2_0_cons_cons_lock"}
//CHECK:     %16 = AIE.buffer(%3) {sym_name = "link2_1_cons_buff_0"} : memref<16xi32>
//CHECK:     %17 = AIE.buffer(%3) {sym_name = "link2_1_cons_buff_1"} : memref<16xi32>
//CHECK:     %18 = AIE.buffer(%3) {sym_name = "link2_1_cons_buff_2"} : memref<16xi32>
//CHECK:     %19 = AIE.lock(%3, 0) {init = 3 : i32, sym_name = "link2_1_cons_prod_lock"}
//CHECK:     %20 = AIE.lock(%3, 1) {init = 0 : i32, sym_name = "link2_1_cons_cons_lock"}
//CHECK:     %21 = AIE.buffer(%1) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
//CHECK:     %22 = AIE.buffer(%1) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
//CHECK:     %23 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "link1_cons_prod_lock"}
//CHECK:     %24 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
//CHECK:     %25 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "link1_prod_lock"}
//CHECK:     %26 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
//CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %3, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %2, DMA : 0)
//CHECK:     AIE.flow(%2, DMA : 0, %3, DMA : 1)
//CHECK:     AIE.shimDMAAllocation @link1(MM2S, 0, 2)
//CHECK:     %27 = AIE.memTileDMA(%1) {
//CHECK:       %30 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%23, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%21 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%24, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%23, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%22 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%24, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       %31 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
//CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
//CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%21 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%23, Release, 1)
//CHECK:       AIE.nextBd ^bb5
//CHECK:     ^bb5:  // pred: ^bb4
//CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%22 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%23, Release, 1)
//CHECK:       AIE.nextBd ^bb4
//CHECK:     ^bb6:  // pred: ^bb3
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %28 = AIE.mem(%2) {
//CHECK:       %30 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%12 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%15, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%13 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%15, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       %31 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
//CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
//CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%8 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%10, Release, 1)
//CHECK:       AIE.nextBd ^bb5
//CHECK:     ^bb5:  // pred: ^bb4
//CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%9 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%10, Release, 1)
//CHECK:       AIE.nextBd ^bb4
//CHECK:     ^bb6:  // pred: ^bb3
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %29 = AIE.mem(%3) {
//CHECK:       %30 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
//CHECK:       AIE.useLock(%19, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%16 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%20, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%19, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%17 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%20, Release, 1)
//CHECK:       AIE.nextBd ^bb3
//CHECK:     ^bb3:  // pred: ^bb2
//CHECK:       AIE.useLock(%19, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%18 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%20, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb4:  // pred: ^bb0
//CHECK:       %31 = AIE.dmaStart(S2MM, 1, ^bb5, ^bb7)
//CHECK:     ^bb5:  // 2 preds: ^bb4, ^bb6
//CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.nextBd ^bb6
//CHECK:     ^bb6:  // pred: ^bb5
//CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%5 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.nextBd ^bb5
//CHECK:     ^bb7:  // pred: ^bb4
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }

module @link_broadcast {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)
        %tile33 = AIE.tile(3, 3)

        AIE.objectFifo @link1 (%tile20, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<48xi32>>
        AIE.objectFifo @link2 (%tile21, {%tile22, %tile33}, [2, 2, 3]) : !AIE.objectFifo<memref<16xi32>>

        AIE.objectFifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>

        AIE.objectFifo.link [@link1] -> [@link2] ()
    }
}
