//===- link_via_shared_mem_padDimensions.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// ObjectFifoLink where output has padDimensions and a larger element type
// than the input. MemTile buffers should use the input (smaller) size since
// padding is applied on-the-fly by the DMA during MM2S transfer.

// CHECK: %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK: %{{.*}}tile_0_1 = aie.tile(0, 1)
// CHECK: %{{.*}}tile_0_2 = aie.tile(0, 2)

// Compute tile buffers use the output (padded) size
// CHECK: %[[OUT_BUF0:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "of_out_cons_buff_0"} : memref<512xi32>
// CHECK: %[[OUT_BUF1:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "of_out_cons_buff_1"} : memref<512xi32>
// CHECK: %[[OUT_PROD:.*]] = aie.lock(%{{.*}}tile_0_2, 0) {init = 2 : i32, sym_name = "of_out_cons_prod_lock_0"}
// CHECK: %[[OUT_CONS:.*]] = aie.lock(%{{.*}}tile_0_2, 1) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}

// MemTile buffers use the input (smaller) size — NOT the output size
// CHECK: %[[MT_BUF0:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "of_in_cons_buff_0"} : memref<256xi32>
// CHECK: %[[MT_BUF1:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "of_in_cons_buff_1"} : memref<256xi32>
// CHECK: %[[MT_PROD:.*]] = aie.lock(%{{.*}}tile_0_1, 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock_0"}
// CHECK: %[[MT_CONS:.*]] = aie.lock(%{{.*}}tile_0_1, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}

// CHECK: aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_1, DMA : 0)
// CHECK: aie.flow(%{{.*}}tile_0_1, DMA : 0, %{{.*}}tile_0_2, DMA : 0)

// MemTile DMA: S2MM receives 256 elements (input size)
// CHECK: %memtile_dma_0_1 = aie.memtile_dma(%{{.*}}tile_0_1) {
// CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[MT_PROD]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[MT_BUF0]] : memref<256xi32>, 0, 256)
// CHECK:   aie.use_lock(%[[MT_CONS]], Release, 1)
// CHECK:   aie.next_bd ^bb2
// CHECK: ^bb2:
// CHECK:   aie.use_lock(%[[MT_PROD]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[MT_BUF1]] : memref<256xi32>, 0, 256)
// CHECK:   aie.use_lock(%[[MT_CONS]], Release, 1)
// CHECK:   aie.next_bd ^bb1

// MemTile DMA: MM2S sends 512 elements with padding (output size)
// CHECK: ^bb3:
// CHECK:   %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK: ^bb4:
// CHECK:   aie.use_lock(%[[MT_CONS]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[MT_BUF0]] : memref<256xi32>, 0, 512, [<size = 64, stride = 4>, <size = 4, stride = 1>], [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>])
// CHECK:   aie.use_lock(%[[MT_PROD]], Release, 1)
// CHECK:   aie.next_bd ^bb5
// CHECK: ^bb5:
// CHECK:   aie.use_lock(%[[MT_CONS]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[MT_BUF1]] : memref<256xi32>, 0, 512, [<size = 64, stride = 4>, <size = 4, stride = 1>], [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>])
// CHECK:   aie.use_lock(%[[MT_PROD]], Release, 1)
// CHECK:   aie.next_bd ^bb4
// CHECK: ^bb6:
// CHECK:   aie.end
// CHECK: }

// Compute tile DMA: S2MM receives 512 elements (full padded size)
// CHECK: %mem_0_2 = aie.mem(%{{.*}}tile_0_2) {
// CHECK:   %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.use_lock(%[[OUT_PROD]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[OUT_BUF0]] : memref<512xi32>, 0, 512)
// CHECK:   aie.use_lock(%[[OUT_CONS]], Release, 1)
// CHECK:   aie.next_bd ^bb2
// CHECK: ^bb2:
// CHECK:   aie.use_lock(%[[OUT_PROD]], AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd(%[[OUT_BUF1]] : memref<512xi32>, 0, 512)
// CHECK:   aie.use_lock(%[[OUT_CONS]], Release, 1)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:
// CHECK:   aie.end
// CHECK: }

module @link_padDimensions_size_mismatch {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of_out(%tile_0_1 dimensionsToStream [<size = 64, stride = 4>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>]>} : !aie.objectfifo<memref<512xi32>>

    aie.objectfifo.link [@of_in] -> [@of_out] ([] [])
  }
}
