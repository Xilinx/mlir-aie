//===- link_via_shared_mem_padDimensions.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// ObjectFifoLink where output has padDimensions and a larger element type
// than the input. MemTile buffers should use the input (smaller) size since
// padding is applied on-the-fly by the DMA during MM2S transfer.

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_in_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_in_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]]) {init = 2 : i32, sym_name = "of_in_cons_prod_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]]) {init = 0 : i32, sym_name = "of_in_cons_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of_out_cons_buff_0"} : memref<512xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_6]]) {sym_name = "of_out_cons_buff_1"} : memref<512xi32>
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_6]]) {init = 2 : i32, sym_name = "of_out_cons_prod_lock_0"}
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_6]]) {init = 0 : i32, sym_name = "of_out_cons_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_6]], DMA : 0)
// CHECK:           aie.shim_dma_allocation @of_in_shim_alloc(%[[VAL_0]], MM2S, 0)
// CHECK:           %[[VAL_11:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<256xi32> offset = 0 len = 256)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<256xi32> offset = 0 len = 512 sizes = [64, 4] strides = [4, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>])
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_12]])
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<256xi32> offset = 0 len = 512 sizes = [64, 4] strides = [4, 1] pad [<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>])
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_12]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.mem(%[[VAL_6]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<512xi32> offset = 0 len = 512)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, %[[VAL_16]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<512xi32> offset = 0 len = 512)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_16]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
