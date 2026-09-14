//===- scratchpad_offset_alignment_warning.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A runtime offset_parameter is an ELEMENT count scaled by the element size,
// and the firmware masks the BD address register with 0xFFFFFFFC. On an
// element type narrower than 4 bytes the product can be rounded down instead
// of rejected; the static offset path rejects the same misalignment outright
// (NpuDmaMemcpyNdOp::verify, "Offset must be 4-byte-aligned"). Warn where it
// can bite, stay silent where it cannot.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file \
// RUN:   --pass-pipeline='any(aie-lower-scratchpad-parameters,aie.device(aie-dma-to-npu))' %s

// 2-byte elements: a byte offset is 4-byte aligned only for an even element
// count, so the value has to be a multiple of 2 elements.
module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @narrow_i16(%in: memref<64xi16>) {
      // expected-warning @+1 {{multiple of 2 elements is silently rounded down}}
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi16>
    }
  }
}

// -----

// 1-byte elements: the value has to be a multiple of 4 elements.
module {
  aiex.scratchpad_parameter @off : i32
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @narrow_i8(%in: memref<64xi8>) {
      // expected-warning @+1 {{multiple of 4 elements is silently rounded down}}
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
        {id = 0 : i64, metadata = @a, offset_parameter = @off} : memref<64xi8>
    }
  }
}

// -----

// 4-byte elements: the product is always 4-byte aligned, so no diagnostic.
// --verify-diagnostics fails the test if an unexpected warning appears here.
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
