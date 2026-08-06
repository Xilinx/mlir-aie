//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On-device check of the ObjectFifo constant-pad VALUE path: a memtile output
// objectfifo with padDimensions + pad_value pads a 13-element i32 transfer up to
// 16 elements (2 before, 1 after) and fills the pad with pad_value = 42. Pure
// DMA passthrough (no core), so the read-back directly exposes the pad fill.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @in0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<13xi32>>
    aie.objectfifo @out0(%tile_0_1 dimensionsToStream [<size = 13, stride = 1>], {%tile_0_0}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 1>]>, pad_value = 42 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out0] ([] [])
    aie.runtime_sequence(%arg0: memref<13xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 13][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<13xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 1 : i64, metadata = @out0, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
