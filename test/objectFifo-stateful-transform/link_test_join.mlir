//===- link_test_join.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 30th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @link_join {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(2, 0)
// CHECK:     %1 = AIE.tile(2, 1)
// CHECK:     %2 = AIE.tile(1, 2)
// CHECK:     %3 = AIE.tile(2, 2)
// CHECK:     %4 = AIE.tile(2, 3)
// CHECK:     %5 = AIE.tile(3, 3)
// CHECK:     AIE.flow(%2, DMA : 0, %1, DMA : 0)
// CHECK:     %6 = AIE.buffer(%2) {sym_name = "link1_buff_0"} : memref<128xi8>
// CHECK:     %7 = AIE.buffer(%2) {sym_name = "link1_buff_1"} : memref<128xi8>
// CHECK:     %8 = AIE.lock(%2, 0) {init = 2 : i32, sym_name = "link1_prod_lock"}
// CHECK:     %9 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:     AIE.flow(%3, DMA : 0, %1, DMA : 1)
// CHECK:     %10 = AIE.buffer(%3) {sym_name = "link2_buff_0"} : memref<128xi8>
// CHECK:     %11 = AIE.buffer(%3) {sym_name = "link2_buff_1"} : memref<128xi8>
// CHECK:     %12 = AIE.lock(%3, 0) {init = 2 : i32, sym_name = "link2_prod_lock"}
// CHECK:     %13 = AIE.lock(%3, 1) {init = 0 : i32, sym_name = "link2_cons_lock"}
// CHECK:     AIE.flow(%4, DMA : 0, %1, DMA : 2)
// CHECK:     %14 = AIE.buffer(%4) {sym_name = "link3_buff_0"} : memref<128xi8>
// CHECK:     %15 = AIE.buffer(%4) {sym_name = "link3_buff_1"} : memref<128xi8>
// CHECK:     %16 = AIE.lock(%4, 0) {init = 2 : i32, sym_name = "link3_prod_lock"}
// CHECK:     %17 = AIE.lock(%4, 1) {init = 0 : i32, sym_name = "link3_cons_lock"}
// CHECK:     AIE.flow(%5, DMA : 0, %1, DMA : 3)
// CHECK:     %18 = AIE.buffer(%5) {sym_name = "link4_buff_0"} : memref<128xi8>
// CHECK:     %19 = AIE.buffer(%5) {sym_name = "link4_buff_1"} : memref<128xi8>
// CHECK:     %20 = AIE.lock(%5, 0) {init = 2 : i32, sym_name = "link4_prod_lock"}
// CHECK:     %21 = AIE.lock(%5, 1) {init = 0 : i32, sym_name = "link4_cons_lock"}
// CHECK:     AIE.flow(%1, DMA : 0, %0, DMA : 0)
// CHECK:     %22 = AIE.buffer(%1) {sym_name = "link5_buff_0"} : memref<512xi8>
// CHECK:     %23 = AIE.buffer(%1) {sym_name = "link5_buff_1"} : memref<512xi8>
// CHECK:     %24 = AIE.lock(%1, 0) {init = 8 : i32, sym_name = "link5_prod_lock"}
// CHECK:     %25 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "link5_cons_lock"}
// CHECK:     %26 = AIE.lock(%0, 0) {init = 1 : i32, sym_name = "link5_cons_prod_lock"}
// CHECK:     %27 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "link5_cons_cons_lock"}
// CHECK:     %28 = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<512xi8>
// CHECK:     %29 = AIE.mem(%2) {
// CHECK:       %35 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%9, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%6 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%8, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%9, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%8, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %30 = AIE.memTileDMA(%1) {
// CHECK:       %35 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%22 : memref<512xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%23 : memref<512xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %36 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%22 : memref<512xi8>, 128, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%23 : memref<512xi8>, 128, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       %37 = AIE.dmaStart(S2MM, 2, ^bb7, ^bb9)
// CHECK:     ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%22 : memref<512xi8>, 256, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%23 : memref<512xi8>, 256, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb7
// CHECK:     ^bb9:  // pred: ^bb6
// CHECK:       %38 = AIE.dmaStart(S2MM, 3, ^bb10, ^bb12)
// CHECK:     ^bb10:  // 2 preds: ^bb9, ^bb11
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%22 : memref<512xi8>, 384, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       AIE.useLock(%24, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%23 : memref<512xi8>, 384, 128>, 0)
// CHECK:       AIE.useLock(%25, Release, 1)
// CHECK:       AIE.nextBd ^bb10
// CHECK:     ^bb12:  // pred: ^bb9
// CHECK:       %39 = AIE.dmaStart(MM2S, 0, ^bb13, ^bb15)
// CHECK:     ^bb13:  // 2 preds: ^bb12, ^bb14
// CHECK:       AIE.useLock(%25, AcquireGreaterEqual, 4)
// CHECK:       AIE.dmaBd(<%22 : memref<512xi8>, 0, 512>, 0)
// CHECK:       AIE.useLock(%24, Release, 4)
// CHECK:       AIE.nextBd ^bb14
// CHECK:     ^bb14:  // pred: ^bb13
// CHECK:       AIE.useLock(%25, AcquireGreaterEqual, 4)
// CHECK:       AIE.dmaBd(<%23 : memref<512xi8>, 0, 512>, 0)
// CHECK:       AIE.useLock(%24, Release, 4)
// CHECK:       AIE.nextBd ^bb13
// CHECK:     ^bb15:  // pred: ^bb12
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %31 = AIE.mem(%3) {
// CHECK:       %35 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%10 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%12, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%13, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%11 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%12, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %32 = AIE.mem(%4) {
// CHECK:       %35 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%17, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%14 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%16, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%17, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%15 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%16, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %33 = AIE.mem(%5) {
// CHECK:       %35 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%21, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%18 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%20, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%21, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%19 : memref<128xi8>, 0, 128>, 0)
// CHECK:       AIE.useLock(%20, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     AIE.shimDMAAllocation @link5(S2MM, 0, 2)
// CHECK:     %34 = AIE.shimDMA(%0) {
// CHECK:       %35 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       AIE.useLock(%26, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%28 : memref<512xi8>, 0, 512>, 0)
// CHECK:       AIE.useLock(%27, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }
     
module @link_join {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile12 = AIE.tile(1, 2)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %tile33 = AIE.tile(3, 3)

        AIE.objectFifo @link1 (%tile12, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<128xi8>>
        AIE.objectFifo @link2 (%tile22, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<128xi8>>
        AIE.objectFifo @link3 (%tile23, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<128xi8>>
        AIE.objectFifo @link4 (%tile33, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<128xi8>>
        AIE.objectFifo @link5 (%tile21, {%tile20}, 2 : i32) : !AIE.objectFifo<memref<512xi8>>

        %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<512xi8>
        AIE.objectFifo.registerExternalBuffers @link5 (%tile20, {%ext_buffer_in}) : (memref<512xi8>)

        AIE.objectFifo.link [@link1, @link2, @link3, @link4] -> [@link5] ()
    }
}
