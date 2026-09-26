//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// Rejected cases for the dynamic (runtime SSA size/stride/len) dma_task path.
// The dynamic BD-word encoder covers every tile type that has a BD register
// layout -- shim NOC, mem tile and core tile -- but not padding. The positive
// cases live in runtime-len.mlir, runtime-len-memtile.mlir and
// runtime-stride-field-widths.mlir.

// Zero padding is a mem tile MM2S feature. Its fields are always compile-time
// constants, but the dynamic packer does not write them, so the combination is
// rejected rather than silently dropping the padding.
module {
  aie.device(npu2) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) {sym_name = "b", address = 4096 : i32} : memref<4096xi8>
    aie.runtime_sequence(%len: i32) {
      %t = aiex.dma_configure_task(%tile, MM2S, 0) {
          // expected-error@+1 {{zero padding is not supported with runtime sizes/strides/len}}
          aie.dma_bd(%buf : memref<4096xi8> offset = 0 len = %len sizes = [2, 2, 4] strides = [4, 8, 1] pad [<const_pad_before=2, const_pad_after=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

// -----

// A constant innermost stride whose byte extent isn't a whole granule is not
// realizable (int8, stride 2 = 16 bits vs the 32-bit granule). A runtime len
// forces the dynamic path. (A unit or granule-aligned innermost stride, runtime
// or constant, is fine -- see runtime-len.mlir.)
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi8>, %len: i32) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{stride 0 is 2 elements at 1 bytes each, not a multiple of the 4-byte address-gen granule}}
          aie.dma_bd(%arg0 : memref<4096xi8> offset = 0 len = %len sizes = [1, 8, 16, 4] strides = [4096, 512, 4, 2]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

// -----

// A constant zero stride paired with a RUNTIME size. The size being unknown is
// exactly why this cannot be waved through: the encoder scales a stride to
// `stride * elemWidth / granularity - 1`, so zero becomes -1 and the packer
// masks it into an all-ones step field. On hardware that walks the BD out of
// its buffer and hangs the channel (observed on a mem tile, NPU1), rather than
// behaving like the "repeat the same block" the stride suggests.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {sym_name = "b2", address = 0 : i32} : memref<4096xi32>
    aie.runtime_sequence(%len: i32, %n: i64) {
      %t = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
          // expected-error@+1 {{stride 1 must be positive unless its size is statically 1}}
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 1, %n, 512] strides = [0, 0, 0, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
