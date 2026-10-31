//===- scratchpad_offset_multiple.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A runtime offset_parameter is an ELEMENT count scaled by the element size,
// and the BD address register drops its low two bits. On an element type
// narrower than 4 bytes the parameter records the element count whose bytes
// fill a word, which the host library makes its value a multiple of, as the
// static offset path rejects the same misalignment (NpuDmaMemcpyNdOp::verify,
// "Offset must be 4-byte-aligned").
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file \
// RUN:   --pass-pipeline='any(aie-lower-scratchpad-parameters,aie.device(aie-dma-to-npu))' %s \
// RUN:   | FileCheck %s

// 2-byte elements: a byte offset is 4-byte aligned only for an even element
// count.
// CHECK-LABEL: aiex.scratchpad_parameter @off : i32
// CHECK-SAME: offset_multiple = 2 : i32
module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @narrow_i16(%in: memref<64xi16>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi16>
    }
  }
}

// -----

// 1-byte elements: four to a word.
// CHECK-LABEL: aiex.scratchpad_parameter @off : i32
// CHECK-SAME: offset_multiple = 4 : i32
module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @narrow_i8(%in: memref<64xi8>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi8>
    }
  }
}

// -----

// 4-byte elements: every element count lands on a word, so no multiple.
// CHECK-LABEL: aiex.scratchpad_parameter @off : i32
// CHECK-NOT: offset_multiple
// CHECK: aie.device
module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @wide_i32(%in: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi32>
    }
  }
}

// -----

// One parameter offsetting both a 2-byte and a 1-byte transfer must suit
// both.
// CHECK-LABEL: aiex.scratchpad_parameter @off : i32
// CHECK-SAME: offset_multiple = 4 : i32
module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.shim_dma_allocation @b (%t, MM2S, 1)
    aie.runtime_sequence @mixed(%in: memref<64xi16>, %bytes: memref<64xi8>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,32][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi16>
      aiex.npu.dma_memcpy_nd(%bytes[0,0,0,0][1,1,1,32][0,0,0,1])
        {id = 1 : i64, metadata = @b, offset_parameter = @off} : memref<64xi8>
    }
  }
}
