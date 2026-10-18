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
// another dimension is runtime. (A unit stride is fine -- see
// dma_to_npu_dynamic.mlir.)
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

// A granule-aligned innermost stride is still unrealizable for a sub-word
// element: the DMA steps whole granules, so stride 2 on bf16 would move both
// halves of each word rather than every other element.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xbf16>, %n: i64) {
      // expected-error@+1 {{stride 0 is 2 elements, but must be 1 for 2-byte elements: the DMA moves whole 4-byte granules.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 4][0, 0, 8, 2]) {id = 0 : i64, metadata = @a} : memref<64xbf16>
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

// A guard whose condition folds to false is a compile-time error rather than
// a cf.assert that could never pass: the constant d0 extent alone runs past
// the 64-element host buffer, whatever the runtime d1 size is.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xi32>, %n: i64) {
      // expected-error@+2 {{a runtime DMA access runs past the end of its 64-element host buffer}}
      // expected-error@+1 {{failed to legalize operation 'aiex.npu.dma_memcpy_nd'}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 128][0, 0, 128, 1]) {id = 0 : i64, metadata = @a} : memref<64xi32>
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

// A constant offset past the end of the host buffer is likewise rejected at
// compile time when the walk has a runtime size.
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @s(%arg0: memref<64xi32>, %n: i64) {
      // expected-error@+2 {{a runtime DMA access runs past the end of its 64-element host buffer}}
      // expected-error@+1 {{failed to legalize operation 'aiex.npu.dma_memcpy_nd'}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 64][1, 1, 1, %n][0, 0, 0, 1]) {id = 0 : i64, metadata = @a} : memref<64xi32>
    }
  }
}
