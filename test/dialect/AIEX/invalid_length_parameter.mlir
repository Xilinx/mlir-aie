//===- invalid_length_parameter.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter requires length_unit}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64) {length_parameter = @n}
      aie.end
    }
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_unit must be a multiple of 16 bytes, got 8 bytes}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64) {length_parameter = @n, length_unit = 2 : i32}
      aie.end
    }
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op length_unit (4611686018427387904 elements) exceeds the 32-bit word count of a BD length}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma, length_parameter = @n, length_unit = 4611686018427387904 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op length_unit (4294967296 elements) exceeds the 32-bit word count of a BD length}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma, length_parameter = @n, length_unit = 4294967296 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter requires a static length that is a multiple of 16 bytes, got 24 bytes}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 6) {length_parameter = @n, length_unit = 4 : i32}
      aie.end
    }
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi4>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter requires a whole-byte element type}}
      aie.dma_bd(%arg0 : memref<4096xi4> offset = 0 len = 64) {length_parameter = @n, length_unit = 32 : i32}
      aie.end
    }
  }
}

// -----

// The added length continues the third dimension, or the second one moved
// there, and a size of one leaves either without an encoded stride.

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter on a non-contiguous pattern requires its second or third dimension to have a size above one}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 16 sizes = [16] strides = [2]) {length_parameter = @n, length_unit = 16 : i32}
      aie.end
    }
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_unit (12 elements) must be a multiple of the 8 elements moved per step of the dimension the added length continues}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 32 sizes = [4, 8] strides = [16, 1]) {length_parameter = @n, length_unit = 12 : i32}
      aie.end
    }
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_unit (16 elements) must be a multiple of the 32 elements moved per step of the dimension the added length continues}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 64 sizes = [2, 4, 8] strides = [64, 16, 1]) {length_parameter = @n, length_unit = 16 : i32}
      aie.end
    }
  }
}

// -----

// The length is patched by a runtime-sequence instruction, so a BD the device
// configures statically cannot take one.

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<64xi32>
  aie.memtile_dma(%t) {
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
  ^bd0:
    // expected-error @+1 {{'aie.dma_bd' op length_parameter is only supported on a BD in a runtime sequence}}
    aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64) {length_parameter = @n, length_unit = 4 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence(%arg0 : memref<4096xi32>, %len : i64) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op length_parameter requires constant sizes and strides}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, %len][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma, length_parameter = @n, length_unit = 4 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op length_unit (16 elements) must be a multiple of the 32 elements moved per step of the dimension the added length continues}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 4, 8][0, 64, 16, 1]) {id = 0 : i64, metadata = @dma, length_parameter = @n, length_unit = 16 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(xcvc1902) {
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    // expected-error @+1 {{'aiex.npu.dma_memcpy_nd' op length_parameter is only supported on AIE2 and AIE2P devices}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma, length_parameter = @n, length_unit = 4 : i64} : memref<4096xi32>
  }
  %tile = aie.tile(2, 0)
  aie.shim_dma_allocation @dma(%tile, MM2S, 0)
}

// -----

// A fifth dimension needs splitting into several BDs, which would separate the
// runtime length from the dimension it extends.

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<4096xi32>) {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{'aie.dma_bd' op length_parameter requires at most 4 dimensions, got 5}}
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 256 sizes = [2, 2, 2, 4, 8] strides = [1024, 512, 64, 16, 1]) {length_parameter = @n, length_unit = 32 : i32}
      aie.end
    }
  }
}
