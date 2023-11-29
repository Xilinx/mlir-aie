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

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           memref.global "public" @from_memTile_cons : memref<48xi32>
// CHECK:           memref.global "public" @from_memTile : memref<48xi32>
// CHECK:           memref.global "public" @to_memTile_cons : memref<16xi32>
// CHECK:           memref.global "public" @to_memTile : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "from_memTile_cons_prod_lock"}
// CHECK:           %[[VAL_4:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "from_memTile_cons_cons_lock"}
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "from_memTile_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "from_memTile_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "from_memTile_prod_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "from_memTile_cons_lock"}
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "to_memTile_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "to_memTile_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "to_memTile_prod_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "to_memTile_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           %[[VAL_13:.*]] = AIE.external_buffer {sym_name = "ext_buff_in"} : memref<48xi32>
// CHECK:           %[[VAL_14:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_15:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_10]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = AIE.memTileDMA(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_18:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_8]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           AIE.shimDMAAllocation @from_memTile(S2MM, 0, 2)
// CHECK:           %[[VAL_19:.*]] = AIE.shimDMA(%[[VAL_0]]) {
// CHECK:             %[[VAL_20:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             AIE.useLock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_13]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_4]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @link_L1_DDR {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)

        AIE.objectfifo @to_memTile (%tile22, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
        AIE.objectfifo @from_memTile (%tile21, {%tile20}, 2 : i32) : !AIE.objectfifo<memref<48xi32>>

        AIE.objectfifo.link [@to_memTile] -> [@from_memTile] ()

        %ext_buff_in = AIE.external_buffer {sym_name = "ext_buff_in"}: memref<48xi32>
        AIE.objectfifo.register_external_buffers @from_memTile (%tile20, {%ext_buff_in}) : (memref<48xi32>)
    }
}
