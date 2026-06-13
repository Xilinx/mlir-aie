//===- bad_npu_nd.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A non-contiguous strided access with d0 > 1023 must still be rejected.
// sizes=[1,1,1080,1920], strides=[0,0,1921,1]: stride1=1921 != sizes0=1920,
// so this is NOT a contiguous row scan and cannot be exempted.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<1920x1080xi32>, %buf : memref<32xi32>, %out : memref<1920x1080xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c1920 = arith.constant 1920 : i64
      %c1921 = arith.constant 1921 : i64
      %c1080 = arith.constant 1080 : i64
      // expected-error@+1 {{Size 0 exceeds the [0:1023] range}}
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0][%c1,%c1080,%c1920][%c0,%c1921,%c1]) { metadata = @of_fromMem, id = 0 : i64 } : memref<1920x1080xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %c32 = arith.constant 32 : i64
      %c128 = arith.constant 128 : i64
      // expected-error@+1 {{Iteration size exceeds the [1:64] range}}
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0][%c2,%c2,%c8][%c16,%c8,%c1]) { metadata = @of_fromMem, id = 0 : i64, iter_size = 128 : i64, iter_stride = 16 : i64 } : memref<128x4x2x8xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<8388608xi32>, %buf : memref<32xi32>, %out : memref<8388608xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c2097152 = arith.constant 2097152 : i64
      // expected-error@+1 {{Stride 1 exceeds the [1:1048576] range}}
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0][%c1,%c2,%c2][%c0,%c2097152,%c1]) { metadata = @of_fromMem, id = 0 : i64 } : memref<8388608xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
  }
}

// -----

// Offsets need to be 4-byte aligned.

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c8 = arith.constant 8 : i64
      // expected-error@+1 {{Offset must be 4-byte-aligned}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c1][%c1,%c1,%c8][%c0,%c1,%c1]) { metadata = @fifo, id = 0 : i64 } : memref<8xi8>
    }
  }
}

// -----

// Strides and sizes expressed in types other than i32 should not overflow hardware limitations when converted to 4-byte granularity.
// The following tests check this.
  
module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64 
      %c8 = arith.constant 8 : i64
      %c2048 = arith.constant 2048 : i64
      // Although 2048 exceeds the 0:1023 limit for size 0, since the elements are i8s,
      // this should be a size of 512 in address granularity (4 bytes) and hence pass the test.
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c2,%c2048][%c0,%c4,%c1]) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi8>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi16>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c2048 = arith.constant 2048 : i64
      // expected-error@+1 {{Size 0 exceeds the [0:1023] range}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c2,%c2048][%c0,%c4,%c1]) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// Strides and sizes are expressed at 4-byte-granularity in hardware, but we express them at memref element type granularity.
// The following tests make sure the proper errors are generated when this is not possible.

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64  // Stride of 2 i8s = 2 bytes < 4 byte granularity, should not be possible
      %c8 = arith.constant 8 : i64
      // expected-error@+1 {{Stride 1 is 2 elements * 1 bytes = 2 bytes, which is not divisible by 4}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c2,%c8][%c0,%c2,%c1]) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi8>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      // expected-error@+1 {{2 elements at 1 bytes each equal 2 bytes, which is not divisible by 4}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c4,%c1]) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi8>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// stride of 2 i8 is not ok

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      // expected-error@+1 {{Stride 0 is 2 elements * 1 bytes = 2 bytes, which is not divisible by 4}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c8][%c0,%c0,%c2]) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi8>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// stride of 1 i16 is ok, but not with size of 3xi16

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi16>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c3 = arith.constant 3 : i64
      %c8 = arith.constant 8 : i64
      // expected-error@+1 {{3 elements at 2 bytes each equal 6 bytes, which is not divisible by 4}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c3][%c0,%c0,%c1]) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// An iteration stride is only range-checked when the iteration size > 1; with
// iter_size == 1 (a single pass) the stride is never applied, so any value is OK.

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %c8 = arith.constant 8 : i64
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c0,%c1]) { metadata = @objectfifo, id = 0 : i64, iter_size = 1 : i64, iter_stride = 1572864 : i64 } : memref<8xi32>
      // expected-error@+1 {{Iteration stride exceeds the [1:1048576] range.}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c0,%c1]) { metadata = @objectfifo, id = 1 : i64, iter_size = 2 : i64, iter_stride = 1572864 : i64 } : memref<8xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// packet header id limit

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %c8 = arith.constant 8 : i64
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c0,%c1], packet = <pkt_id = 31, pkt_type = 7>) { metadata = @objectfifo, id = 0 : i64 } : memref<8xi32>
      // expected-error@+1 {{Packet ID field can only hold 5 bits.}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c0,%c1], packet = <pkt_id = 32, pkt_type = 2>) { metadata = @objectfifo, id = 1 : i64 } : memref<8xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// packet header type limit

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %c8 = arith.constant 8 : i64
      // expected-error@+1 {{Packet type field can only hold 3 bits.}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c0,%c1], packet = <pkt_id = 2, pkt_type = 8>) { metadata = @objectfifo, id = 1 : i64 } : memref<8xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}

// -----

// repeat_count (a pure replay) and iter_size (a strided iteration) describe two
// different hardware behaviors and cannot both be set on one transfer.

module {
  aie.device(npu1) {
    aie.runtime_sequence(%a : memref<8xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      // expected-error@+1 {{repeat_count and iter_size are mutually exclusive.}}
      aiex.npu.dma_memcpy_nd (%a[%c0,%c0,%c0][%c1,%c1,%c2][%c0,%c0,%c1]) { metadata = @objectfifo, id = 0 : i64, iter_size = 4 : i64, iter_stride = 2 : i64, repeat_count = 3 : i32 } : memref<8xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @objectfifo (%tile_0_0, MM2S, 0)
  }
}
