//===- base_test_AIE2.mlir --------------------------*- MLIR -*-===//
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

// CHECK: module @ndDMAObjFifoAIE2 {
// CHECK:   AIE.device(xcve2302) {
// CHECK:           memref.global "public" @of1_cons : memref<256xi32>
// CHECK:           memref.global "public" @of1 : memref<256xi32>
// CHECK:           memref.global "public" @of0_cons : memref<256xi32>
// CHECK:           memref.global "public" @of0 : memref<256xi32>
// CHECK:     %[[tile_1_2:.*]] = AIE.tile(1, 2)
// CHECK:     %[[tile_1_3:.*]] = AIE.tile(1, 3)
// CHECK:     %[[tile_3_3:.*]] = AIE.tile(3, 3)
// CHECK:     %[[of1_cons_buff_0:.*]] = AIE.buffer(%[[tile_3_3]]) {sym_name = "of1_cons_buff_0"} : memref<256xi32>
// CHECK:     %[[of1_cons_buff_1:.*]] = AIE.buffer(%[[tile_3_3]]) {sym_name = "of1_cons_buff_1"} : memref<256xi32>
// CHECK:     %[[of1_cons_prod_lock:.*]] = AIE.lock(%[[tile_3_3]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %[[of1_cons_cons_lock:.*]] = AIE.lock(%[[tile_3_3]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %[[of1_buff_0:.*]] = AIE.buffer(%[[tile_1_2]]) {sym_name = "of1_buff_0"} : memref<256xi32>
// CHECK:     %[[of1_buff_1:.*]] = AIE.buffer(%[[tile_1_2]]) {sym_name = "of1_buff_1"} : memref<256xi32>
// CHECK:     %[[of1_prod_lock:.*]] = AIE.lock(%[[tile_1_2]], 2) {init = 2 : i32, sym_name = "of1_prod_lock"}
// CHECK:     %[[of1_cons_lock:.*]] = AIE.lock(%[[tile_1_2]], 3) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:     %[[of0_cons_buff_0:.*]] = AIE.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_0"} : memref<256xi32>
// CHECK:     %[[of0_cons_buff_1:.*]] = AIE.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_1"} : memref<256xi32>
// CHECK:     %[[of0_cons_buff_2:.*]] = AIE.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_2"} : memref<256xi32>
// CHECK:     %[[of0_cons_buff_3:.*]] = AIE.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_3"} : memref<256xi32>
// CHECK:     %[[of0_cons_prod_lock:.*]] = AIE.lock(%[[tile_1_3]], 0) {init = 4 : i32, sym_name = "of0_cons_prod_lock"}
// CHECK:     %[[of0_cons_cons_lock:.*]] = AIE.lock(%[[tile_1_3]], 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock"}
// CHECK:     %[[of0_buff_0:.*]] = AIE.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:     %[[of0_buff_1:.*]] = AIE.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:     %[[of0_buff_2:.*]] = AIE.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_2"} : memref<256xi32>
// CHECK:     %[[of0_buff_3:.*]] = AIE.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_3"} : memref<256xi32>
// CHECK:     %[[of0_prod_lock:.*]] = AIE.lock(%[[tile_1_2]], 0) {init = 4 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %[[of0_cons_lock:.*]] = AIE.lock(%[[tile_1_2]], 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:     AIE.flow(%[[tile_1_2]], DMA : 0, %[[tile_1_3]], DMA : 0)
// CHECK:     AIE.flow(%[[tile_1_2]], DMA : 1, %[[tile_3_3]], DMA : 0)
// CHECK:     %[[VAL_23:.*]] = AIE.mem(%[[tile_1_2]]) {
// CHECK:       %[[VAL_26:.*]] = AIE.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       AIE.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_buff_0]] : memref<256xi32>, 0, 256, [<wrap = 16, step = 1>, <wrap = 16, step = 16>, <wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       AIE.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_buff_1]] : memref<256xi32>, 0, 256, [<wrap = 16, step = 1>, <wrap = 16, step = 16>, <wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:       AIE.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_buff_2]] : memref<256xi32>, 0, 256, [<wrap = 16, step = 1>, <wrap = 16, step = 16>, <wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:       AIE.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_buff_3]] : memref<256xi32>, 0, 256, [<wrap = 16, step = 1>, <wrap = 16, step = 16>, <wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:       %[[VAL_27:.*]] = AIE.dma_start(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:       AIE.use_lock(%[[of1_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of1_buff_0]] : memref<256xi32>, 0, 256, [<wrap = 128, step = 2>])
// CHECK:       AIE.use_lock(%[[of1_prod_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:       AIE.use_lock(%[[of1_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of1_buff_1]] : memref<256xi32>, 0, 256, [<wrap = 128, step = 2>])
// CHECK:       AIE.use_lock(%[[of1_prod_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             AIE.end
// CHECK:           }
// CHECK:     %[[VAL_24:.*]] = AIE.mem(%[[tile_1_3]]) {
// CHECK:       %[[VAL_26:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       AIE.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_cons_buff_0]] : memref<256xi32>, 0, 256, [<wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       AIE.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_cons_buff_1]] : memref<256xi32>, 0, 256, [<wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:       AIE.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_cons_buff_2]] : memref<256xi32>, 0, 256, [<wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:       AIE.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of0_cons_buff_3]] : memref<256xi32>, 0, 256, [<wrap = 1, step = 1>])
// CHECK:       AIE.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:     %[[VAL_25:.*]] = AIE.mem(%[[tile_3_3]]) {
// CHECK:       %[[VAL_26:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.use_lock(%[[of1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of1_cons_buff_0]] : memref<256xi32>, 0, 256)
// CHECK:       AIE.use_lock(%[[of1_cons_cons_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       AIE.use_lock(%[[of1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       AIE.dma_bd(%[[of1_cons_buff_1]] : memref<256xi32>, 0, 256)
// CHECK:       AIE.use_lock(%[[of1_cons_cons_lock]], Release, 1)
// CHECK:             AIE.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @ndDMAObjFifoAIE2 {
 AIE.device(xcve2302) {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)
    %tile33 = AIE.tile(3, 3)

    // Even if an objectFifo could be implemented in shared memory, as with
    // this case between two adjacent tiles, we need to use DMAs if a data
    // layout transformation with toStream and fromStream was specified.
    AIE.objectfifo @of0 (%tile12 toStream [<wrap = 16, step = 1>, <wrap = 16, step = 16>, <wrap = 1, step = 1>], // transpose
                         {%tile13 fromStream [<wrap = 1, step = 1>]},
                         4 : i32) : !AIE.objectfifo<memref<256xi32>>

    AIE.objectfifo @of1 (%tile12 toStream [<wrap = 128, step = 2>], {%tile33},
                         2 : i32) : !AIE.objectfifo<memref<256xi32>>
 }
}
