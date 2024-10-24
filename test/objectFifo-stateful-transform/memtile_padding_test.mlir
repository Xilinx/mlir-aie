//===- memtile_padding_test.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @objFifo_in0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<56x56xi8>>
    aie.objectfifo @objFifo_in1(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo.link [@objFifo_in0] -> [@objFifo_in1] ([] [])
    aie.objectfifo @objFifo_out1(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo @objFifo_out0(%tile_0_1 dimensionsToStream [<size = 61, stride = 56>, <size = 56, stride = 1>], {%tile_0_0}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 2, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 4>]>} : !aie.objectfifo<memref<64x64xi8>>
    aie.objectfifo.link [@objFifo_out1] -> [@objFifo_out0] ([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %subview = aie.objectfifo.acquire @objFifo_in1 (Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
      %subview1 = aie.objectfifo.acquire @objFifo_out1 (Produce, 1) : !aie.objectfifosubview<memref<64x64xi8>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
      %elem1 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
      
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c12_i8 = arith.constant 12 : i8
      scf.for %arg1 = %c0 to %c64 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %0 = memref.load %elem[%arg1, %arg2] : memref<64x64xi8>
          %1 = arith.addi %0, %c12_i8 : i8
          memref.store %1, %elem1[%arg1, %arg2] : memref<64x64xi8>
        }
      }
      aie.objectfifo.release @objFifo_in1 (Consume, 1)
      aie.objectfifo.release @objFifo_out1 (Produce, 1)
      aie.end
    }

    aiex.runtime_sequence(%arg0: memref<61x56xi8>, %arg1: memref<32xi8>, %arg2: memref<64x64xi8>) {
      aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 61, 56][0, 0, 56, 1]) {id = 0 : i64, metadata = @objFifo_in0} : memref<61x56xi8>
      aiex.npu.dma_memcpy_nd (0, 0, %arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64, 1]) {id = 1 : i64, metadata = @objFifo_out0, issue_token = true} : memref<64x64xi8>
      aiex.npu.dma_wait { symbol = @objFifo_out0 }
    }
  }
}