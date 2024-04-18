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

module {
  aie.device(npu) {
    func.func @bad_npu_nd_length(%in : memref<1920x1080xi32>, %buf : memref<32xi32>, %out : memref<1920x1080xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c1920 = arith.constant 1920 : i64
      %c1080 = arith.constant 1080 : i64
      // expected-error@+1 {{Size 0 exceeds the [0:1023] range}}
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1080,%c1920][%c0,%c0,%c1920]) { metadata = @of_fromMem, id = 0 : i64 } : memref<1920x1080xi32>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_npu_nd_repeat(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %c32 = arith.constant 32 : i64
      %c128 = arith.constant 128 : i64
      // expected-error@+1 {{Size 3 exceeds the [1:64] range}}
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c128,%c2,%c2,%c8][%c0,%c16,%c8]) { metadata = @of_fromMem, id = 0 : i64 } : memref<128x4x2x8xi32>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_npu_nd_stride(%in : memref<8388608xi32>, %buf : memref<32xi32>, %out : memref<8388608xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c2097152 = arith.constant 2097152 : i64
      // expected-error@+1 {{Stride 1 exceeds the [1:1M] range}}
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c2,%c2][%c0,%c0,%c2097152]) { metadata = @of_fromMem, id = 0 : i64 } : memref<8388608xi32>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_npu_nd_type(%in : memref<1920x1080xi8>, %buf : memref<32xi32>, %out : memref<1920x1080xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c1920 = arith.constant 1920 : i64
      %c1080 = arith.constant 1080 : i64
      // expected-error@+1 {{must be used with memref type with element width 32.}}
      aiex.npu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1080,%c1920][%c0,%c0,%c1920]) { metadata = @of_fromMem, id = 0 : i64 } : memref<1920x1080xi8>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}
