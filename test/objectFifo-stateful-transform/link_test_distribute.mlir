//===- link_test_distribute.mlir ------------------------------------------------*- MLIR -*-===//
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

//CHECK: module @link_distribute {
//CHECK:   AIE.device(xcve2302) {
//CHECK:     memref.global "public" @link4_cons : memref<12xi32>
//CHECK:     memref.global "public" @link4 : memref<12xi32>
//CHECK:     memref.global "public" @link3_cons : memref<20xi32>
//CHECK:     memref.global "public" @link3 : memref<20xi32>
//CHECK:     memref.global "public" @link2_cons : memref<4x4xi32>
//CHECK:     memref.global "public" @link2 : memref<4x4xi32>
//CHECK:     memref.global "public" @link1_cons : memref<48xi32>
//CHECK:     memref.global "public" @link1 : memref<48xi32>
//CHECK:     %0 = AIE.tile(2, 0)
//CHECK:     %1 = AIE.tile(2, 1)
//CHECK:     %2 = AIE.tile(2, 2)
//CHECK:     %3 = AIE.tile(2, 3)
//CHECK:     %4 = AIE.tile(3, 3)
//CHECK:     %5 = AIE.buffer(%4) {sym_name = "link4_cons_buff_0"} : memref<12xi32>
//CHECK:     %6 = AIE.buffer(%4) {sym_name = "link4_cons_buff_1"} : memref<12xi32>
//CHECK:     %7 = AIE.lock(%4, 0) {init = 2 : i32, sym_name = "link4_cons_prod_lock"}
//CHECK:     %8 = AIE.lock(%4, 1) {init = 0 : i32, sym_name = "link4_cons_cons_lock"}
//CHECK:     %9 = AIE.buffer(%3) {sym_name = "link3_cons_buff_0"} : memref<20xi32>
//CHECK:     %10 = AIE.buffer(%3) {sym_name = "link3_cons_buff_1"} : memref<20xi32>
//CHECK:     %11 = AIE.lock(%3, 0) {init = 2 : i32, sym_name = "link3_cons_prod_lock"}
//CHECK:     %12 = AIE.lock(%3, 1) {init = 0 : i32, sym_name = "link3_cons_cons_lock"}
//CHECK:     %13 = AIE.buffer(%2) {sym_name = "link2_cons_buff_0"} : memref<4x4xi32>
//CHECK:     %14 = AIE.buffer(%2) {sym_name = "link2_cons_buff_1"} : memref<4x4xi32>
//CHECK:     %15 = AIE.lock(%2, 0) {init = 2 : i32, sym_name = "link2_cons_prod_lock"}
//CHECK:     %16 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "link2_cons_cons_lock"}
//CHECK:     %17 = AIE.buffer(%1) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
//CHECK:     %18 = AIE.buffer(%1) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
//CHECK:     %19 = AIE.lock(%1, 0) {init = 6 : i32, sym_name = "link1_cons_prod_lock"}
//CHECK:     %20 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
//CHECK:     %21 = AIE.lock(%0, 0) {init = 1 : i32, sym_name = "link1_prod_lock"}
//CHECK:     %22 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
//CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %2, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 1, %3, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 2, %4, DMA : 0)
//CHECK:     %23 = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<48xi32>
//CHECK:     AIE.shimDMAAllocation @link1(MM2S, 0, 2)
//CHECK:     %24 = AIE.shimDMA(%0) {
//CHECK:       %29 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
//CHECK:       AIE.useLock(%22, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%23 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%21, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb2:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %25 = AIE.memTileDMA(%1) {
//CHECK:       %29 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%19, AcquireGreaterEqual, 3)
//CHECK:       AIE.dmaBd(<%17 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%20, Release, 3)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%19, AcquireGreaterEqual, 3)
//CHECK:       AIE.dmaBd(<%18 : memref<48xi32>, 0, 48>, 0)
//CHECK:       AIE.useLock(%20, Release, 3)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       %30 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
//CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
//CHECK:       AIE.useLock(%20, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%17 : memref<48xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb5
//CHECK:     ^bb5:  // pred: ^bb4
//CHECK:       AIE.useLock(%20, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%18 : memref<48xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb4
//CHECK:     ^bb6:  // pred: ^bb3
//CHECK:       %31 = AIE.dmaStart(MM2S, 1, ^bb7, ^bb9)
//CHECK:     ^bb7:  // 2 preds: ^bb6, ^bb8
//CHECK:       AIE.useLock(%20, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%17 : memref<48xi32>, 64, 20>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb8
//CHECK:     ^bb8:  // pred: ^bb7
//CHECK:       AIE.useLock(%20, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%18 : memref<48xi32>, 64, 20>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb7
//CHECK:     ^bb9:  // pred: ^bb6
//CHECK:       %32 = AIE.dmaStart(MM2S, 2, ^bb10, ^bb12)
//CHECK:     ^bb10:  // 2 preds: ^bb9, ^bb11
//CHECK:       AIE.useLock(%20, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%17 : memref<48xi32>, 144, 12>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb11
//CHECK:     ^bb11:  // pred: ^bb10
//CHECK:       AIE.useLock(%20, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%18 : memref<48xi32>, 144, 12>, 0)
//CHECK:       AIE.useLock(%19, Release, 1)
//CHECK:       AIE.nextBd ^bb10
//CHECK:     ^bb12:  // pred: ^bb9
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %26 = AIE.mem(%2) {
//CHECK:       %29 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%15, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%13 : memref<4x4xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%16, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%15, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%14 : memref<4x4xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%16, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %27 = AIE.mem(%3) {
//CHECK:       %29 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%9 : memref<20xi32>, 0, 20>, 0)
//CHECK:       AIE.useLock(%12, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%10 : memref<20xi32>, 0, 20>, 0)
//CHECK:       AIE.useLock(%12, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %28 = AIE.mem(%4) {
//CHECK:       %29 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%7, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%5 : memref<12xi32>, 0, 12>, 0)
//CHECK:       AIE.useLock(%8, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%7, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%6 : memref<12xi32>, 0, 12>, 0)
//CHECK:       AIE.useLock(%8, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }
     
module @link_distribute {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %tile33 = AIE.tile(3, 3)

        AIE.objectFifo @link1 (%tile20, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<48xi32>>
        AIE.objectFifo @link2 (%tile21, {%tile22}, 2 : i32) : !AIE.objectFifo<memref<4x4xi32>>
        AIE.objectFifo @link3 (%tile21, {%tile23}, 2 : i32) : !AIE.objectFifo<memref<20xi32>>
        AIE.objectFifo @link4 (%tile21, {%tile33}, 2 : i32) : !AIE.objectFifo<memref<12xi32>>

        %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<48xi32>
        AIE.objectFifo.registerExternalBuffers @link1 (%tile20, {%ext_buffer_in}) : (memref<48xi32>)

        AIE.objectFifo.link [@link1] -> [@link2, @link3, @link4] ()
    }
}
