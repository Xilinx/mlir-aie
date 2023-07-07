//===- link_test_L1_to_DDR.mlir ------------------------------------------------*- MLIR -*-===//
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

// CHECK: module @link_L1_DDR {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(2, 0)
// CHECK:     %1 = AIE.tile(2, 1)
// CHECK:     %2 = AIE.tile(2, 2)
// CHECK:     AIE.flow(%2, DMA : 0, %1, DMA : 0)
// CHECK:     %3 = AIE.buffer(%2) {sym_name = "to_memTile_buff_0"} : memref<16xi32>
// CHECK:     %4 = AIE.buffer(%2) {sym_name = "to_memTile_buff_1"} : memref<16xi32>
// CHECK:     %5 = AIE.lock(%2, 0) {init = 2 : i32, sym_name = "to_memTile_prod_lock"}
// CHECK:     %6 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "to_memTile_cons_lock"}
// CHECK:     AIE.flow(%1, DMA : 0, %0, DMA : 0)
// CHECK:     %7 = AIE.buffer(%1) {sym_name = "from_memTile_buff_0"} : memref<48xi32>
// CHECK:     %8 = AIE.buffer(%1) {sym_name = "from_memTile_buff_1"} : memref<48xi32>
// CHECK:     %9 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "from_memTile_prod_lock"}
// CHECK:     %10 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "from_memTile_cons_lock"}
// CHECK:     %11 = AIE.lock(%0, 0) {init = 1 : i32, sym_name = "from_memTile_cons_prod_lock"}
// CHECK:     %12 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "from_memTile_cons_cons_lock"}
// CHECK:     %13 = AIE.external_buffer {sym_name = "ext_buff_in"} : memref<48xi32>
// CHECK:     %14 = AIE.mem(%2) {
// CHECK:       %17 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%3 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%5, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%5, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %15 = AIE.memTileDMA(%1) {
// CHECK:       %17 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%9, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<48xi32>, 0, 48>, 0)
// CHECK:       AIE.useLock(%10, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%9, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%8 : memref<48xi32>, 0, 48>, 0)
// CHECK:       AIE.useLock(%10, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %18 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       AIE.useLock(%10, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<48xi32>, 0, 48>, 0)
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.nextBd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       AIE.useLock(%10, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%8 : memref<48xi32>, 0, 48>, 0)
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.nextBd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %16 = AIE.shimDMA(%0) {
// CHECK:       %17 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%13 : memref<48xi32>, 0, 48>, 0)
// CHECK:       AIE.useLock(%12, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK:   AIE.shimDMAAllocation("from_memTile", S2MM, 0, 2)
// CHECK: }

module @link_L1_DDR {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)

        %objFifo = AIE.objectFifo.createObjectFifo(%tile22, {%tile21}, 2 : i32) {sym_name = "to_memTile"} : !AIE.objectFifo<memref<16xi32>>
        %objFifo2 = AIE.objectFifo.createObjectFifo(%tile21, {%tile20}, 2 : i32) {sym_name = "from_memTile"} : !AIE.objectFifo<memref<48xi32>>

        AIE.objectFifo.link({%objFifo}, {%objFifo2}) : ({!AIE.objectFifo<memref<16xi32>>}, {!AIE.objectFifo<memref<48xi32>>})

        %ext_buff_in = AIE.external_buffer {sym_name = "ext_buff_in"}: memref<48xi32> 
        AIE.objectFifo.registerExternalBuffers(%tile20, %objFifo2 : !AIE.objectFifo<memref<48xi32>>, {%ext_buff_in}) : (memref<48xi32>)
    }
}
