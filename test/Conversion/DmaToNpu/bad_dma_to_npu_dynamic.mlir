//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Rejected cases for dynamic (runtime SSA size/stride) dma_memcpy_nd lowering.

// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s

// Runtime offsets are not supported (the buffer pointer is set by the address
// patch from a compile-time offset).
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<4096xi32>, %n: i64) {
      // expected-error@+1 {{Only constant offsets currently supported}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, %n][2, 4, 8, 32][2048, 256, 64, 1]) {id = 0 : i64, metadata = @a} : memref<4096xi32>
    }
  }
}

// -----

// The innermost stride must be a compile-time constant 1 when any size/stride
// is runtime.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xi32>, %n: i64) {
      // expected-error@+1 {{innermost stride must be a compile-time constant 1}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, %n][0, 0, 4, 2]) {id = 0 : i64, metadata = @a} : memref<64xi32>
    }
  }
}

// -----

// A constant size that exceeds its hardware field is a hard error even when
// another dimension is runtime (d0 wrap is 10-bit).
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<1048576xi32>, %n: i64) {
      // expected-error@+1 {{d0 size constant value 2000 exceeds hardware range [0:1023]}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 4, %n, 2000][100000, 25000, 64, 1]) {id = 0 : i64, metadata = @a} : memref<1048576xi32>
    }
  }
}

// -----

// The assert_bd_field guard op rejects a constant value over its field max.
module {
  aie.device(npu1) {
    aie.runtime_sequence @s(%arg0: memref<4xi32>) {
      %c = arith.constant 5000 : i32
      // expected-error@+1 {{constant value 5000 exceeds the guarded field range [0:1023]}}
      aiex.npu.assert_bd_field(%c) {max = 1023 : i32} : i32
    }
  }
}
