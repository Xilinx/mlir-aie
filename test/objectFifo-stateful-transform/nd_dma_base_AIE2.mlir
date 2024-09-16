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
// CHECK:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @of1_cons : memref<256xi32>
// CHECK:           memref.global "public" @of1 : memref<256xi32>
// CHECK:           memref.global "public" @of0_cons : memref<256xi32>
// CHECK:           memref.global "public" @of0 : memref<256xi32>
// CHECK:     %[[tile_1_2:.*]] = aie.tile(1, 2)
// CHECK:     %[[tile_1_3:.*]] = aie.tile(1, 3)
// CHECK:     %[[tile_3_3:.*]] = aie.tile(3, 3)
// CHECK:     %[[of1_cons_buff_0:.*]] = aie.buffer(%[[tile_3_3]]) {sym_name = "of1_cons_buff_0"} : memref<256xi32>
// CHECK:     %[[of1_cons_buff_1:.*]] = aie.buffer(%[[tile_3_3]]) {sym_name = "of1_cons_buff_1"} : memref<256xi32>
// CHECK:     %[[of1_cons_prod_lock:.*]] = aie.lock(%[[tile_3_3]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %[[of1_cons_cons_lock:.*]] = aie.lock(%[[tile_3_3]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %[[of1_buff_0:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "of1_buff_0"} : memref<256xi32>
// CHECK:     %[[of1_buff_1:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "of1_buff_1"} : memref<256xi32>
// CHECK:     %[[of1_prod_lock:.*]] = aie.lock(%[[tile_1_2]], 2) {init = 2 : i32, sym_name = "of1_prod_lock"}
// CHECK:     %[[of1_cons_lock:.*]] = aie.lock(%[[tile_1_2]], 3) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:     %[[of0_cons_buff_0:.*]] = aie.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_0"} : memref<256xi32>
// CHECK:     %[[of0_cons_buff_1:.*]] = aie.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_1"} : memref<256xi32>
// CHECK:     %[[of0_cons_buff_2:.*]] = aie.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_2"} : memref<256xi32>
// CHECK:     %[[of0_cons_buff_3:.*]] = aie.buffer(%[[tile_1_3]]) {sym_name = "of0_cons_buff_3"} : memref<256xi32>
// CHECK:     %[[of0_cons_prod_lock:.*]] = aie.lock(%[[tile_1_3]], 0) {init = 4 : i32, sym_name = "of0_cons_prod_lock"}
// CHECK:     %[[of0_cons_cons_lock:.*]] = aie.lock(%[[tile_1_3]], 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock"}
// CHECK:     %[[of0_buff_0:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:     %[[of0_buff_1:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:     %[[of0_buff_2:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_2"} : memref<256xi32>
// CHECK:     %[[of0_buff_3:.*]] = aie.buffer(%[[tile_1_2]]) {sym_name = "of0_buff_3"} : memref<256xi32>
// CHECK:     %[[of0_prod_lock:.*]] = aie.lock(%[[tile_1_2]], 0) {init = 4 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %[[of0_cons_lock:.*]] = aie.lock(%[[tile_1_2]], 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:     aie.flow(%[[tile_1_2]], DMA : 0, %[[tile_1_3]], DMA : 0)
// CHECK:     aie.flow(%[[tile_1_2]], DMA : 1, %[[tile_3_3]], DMA : 0)
// CHECK:     %[[VAL_23:.*]] = aie.mem(%[[tile_1_2]]) {
// CHECK:       %[[VAL_26:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_buff_0]] : memref<256xi32>, 0, 256, [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_buff_1]] : memref<256xi32>, 0, 256, [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_buff_2]] : memref<256xi32>, 0, 256, [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[of0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_buff_3]] : memref<256xi32>, 0, 256, [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_prod_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:       %[[VAL_27:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:       aie.use_lock(%[[of1_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of1_buff_0]] : memref<256xi32>, 0, 256, [<size = 128, stride = 2>])
// CHECK:       aie.use_lock(%[[of1_prod_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(%[[of1_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of1_buff_1]] : memref<256xi32>, 0, 256, [<size = 128, stride = 2>])
// CHECK:       aie.use_lock(%[[of1_prod_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK:     %[[VAL_24:.*]] = aie.mem(%[[tile_1_3]]) {
// CHECK:       %[[VAL_26:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_cons_buff_0]] : memref<256xi32>, 0, 256, [<size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_cons_buff_1]] : memref<256xi32>, 0, 256, [<size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_cons_buff_2]] : memref<256xi32>, 0, 256, [<size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[of0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of0_cons_buff_3]] : memref<256xi32>, 0, 256, [<size = 1, stride = 1>])
// CHECK:       aie.use_lock(%[[of0_cons_cons_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:     %[[VAL_25:.*]] = aie.mem(%[[tile_3_3]]) {
// CHECK:       %[[VAL_26:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[of1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of1_cons_buff_0]] : memref<256xi32>, 0, 256)
// CHECK:       aie.use_lock(%[[of1_cons_cons_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[of1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[of1_cons_buff_1]] : memref<256xi32>, 0, 256)
// CHECK:       aie.use_lock(%[[of1_cons_cons_lock]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    // Even if an objectFifo could be implemented in shared memory, as with
    // this case between two adjacent tiles, we need to use DMAs if a data
    // layout transformation with dimensionsToStream and dimensionsFromStream was specified.
    aie.objectfifo @of0 (%tile12 dimensionsToStream [<size = 16, stride = 1>, <size = 16, stride = 16>, <size = 1, stride = 1>], // transpose
                         {%tile13 dimensionsFromStream [<size = 1, stride = 1>]},
                         4 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile12 dimensionsToStream [<size = 128, stride = 2>], {%tile33},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>
 }
}
