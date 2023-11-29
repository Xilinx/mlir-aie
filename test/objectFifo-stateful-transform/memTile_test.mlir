//===- memTile_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Xilinx Inc.
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           memref.global "public" @of_cons : memref<16xi32>
// CHECK:           memref.global "public" @of : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "of_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "of_cons_prod_lock"}
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_cons_cons_lock"}
// CHECK:           %[[VAL_6:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_7:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 2 : i32, sym_name = "of_prod_lock"}
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_10:.*]] = AIE.memTileDMA(%[[VAL_0]]) {
// CHECK:             %[[VAL_11:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_7]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = AIE.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_13:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_2]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_3]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @memTile {
   AIE.device(xcve2302) {
      %tile11 = AIE.tile(2, 1)
      %tile12 = AIE.tile(2, 2)

      AIE.objectfifo @of (%tile11, {%tile12}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
   }
}
