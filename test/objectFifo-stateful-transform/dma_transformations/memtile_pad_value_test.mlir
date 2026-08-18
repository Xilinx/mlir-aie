//===- memtile_pad_value_test.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// An objectfifo carrying both padDimensions and padValue lowers the pad
// GEOMETRY onto the emitted aie.dma_bd and routes the pad VALUE onto the memtile
// MM2S channel op (aie.dma_start) that applies the padding -- matching the
// hardware split (geometry per-BD, value per-MM2S-channel register).

// CHECK: aie.dma_start(MM2S, {{[0-9]+}}, {{.*}}) {pad_value = 7 : i32}
// CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>])

module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @objFifo_in0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<56x56xi8>>
    aie.objectfifo @objFifo_in1(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])
    aie.objectfifo @objFifo_out1(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo @objFifo_out0(%tile_0_1 dimensionsToStream [<size = 61, stride = 56>, <size = 56, stride = 1>], {%tile_0_0}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>]>, padValue = 7 : i32} : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %elem1 = aie.objectfifo.acquire @objFifo_in1(Consume) : memref<64x64xi8>
      %subview1_obj0 = aie.objectfifo.acquire @objFifo_out1(Produce) : memref<64x64xi8>
      aie.objectfifo.release @objFifo_in1(Consume) [1]
      aie.objectfifo.release @objFifo_out1(Produce) [1]
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}
