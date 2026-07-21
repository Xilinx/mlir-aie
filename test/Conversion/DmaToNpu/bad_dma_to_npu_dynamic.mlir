//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Rejected cases for dynamic (runtime SSA size/stride) dma_memcpy_nd lowering.

// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s

// (Runtime offsets ARE supported -- they lower to an arith-built arg_plus on
// the address patch; see dma_to_npu_dynamic.mlir. Only genuinely unrealizable
// or unencodable cases are rejected below.)

// A constant innermost stride whose byte extent isn't a whole granule is not
// realizable (int8, stride 2 = 16 bits vs the 32-bit granule), even when
// another dimension is runtime. (A unit stride, or a granule-aligned one, is
// fine -- see dma_to_npu_dynamic.mlir.)
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xi8>, %n: i64) {
      // expected-error@+1 {{stride 0 is 2 elements at 1 bytes each, not a multiple of the 4-byte address-gen granule}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, %n][0, 0, 4, 2]) {id = 0 : i64, metadata = @a} : memref<64xi8>
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
      // expected-error@+1 {{d0 size hardware value 2000 exceeds hardware range [0:1023]}}
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

// -----

// A constant innermost (d0) size whose byte extent isn't a whole granule is not
// realizable in hardware (int8, size 2 = 16 bits vs the 32-bit granule), even
// when another dimension is runtime.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<4096xi8>, %n: i64) {
      // expected-error@+1 {{d0 size 2 elements at 1 bytes each is not a multiple of the 4-byte address-gen granule}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 2][0, 0, 2, 1]) {id = 0 : i64, metadata = @a} : memref<4096xi8>
    }
  }
}

// -----

// A constant outer stride whose byte extent isn't a whole granule is likewise
// unrealizable (int8, stride 3 = 24 bits vs the 32-bit granule).
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<4096xi8>, %n: i64) {
      // expected-error@+1 {{stride 2 is 3 elements at 1 bytes each, not a multiple of the 4-byte address-gen granule}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 4, %n, 4][0, 3, 4, 1]) {id = 0 : i64, metadata = @a} : memref<4096xi8>
    }
  }
}

// -----

// The assert_bd_divisible guard op rejects a constant value not divisible by
// its divisor.
module {
  aie.device(npu1) {
    aie.runtime_sequence @s(%arg0: memref<4xi32>) {
      %c = arith.constant 3 : i32
      // expected-error@+1 {{constant value 3 is not divisible by 4}}
      aiex.npu.assert_bd_divisible(%c) {divisor = 4 : i32} : i32
    }
  }
}
