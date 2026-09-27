//===- scratchpad_size_parameter_invalid.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The transfers a size_parameter cannot patch. It rewrites the length word of
// one statically written shim BD, in units of the two innermost dimensions.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file \
// RUN:   --aie-lower-scratchpad-parameters --aie-substitute-shim-dma-allocations \
// RUN:   --aie-decompose-large-dma-bd --aie-assign-runtime-sequence-bd-ids \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s

// The state table word the firmware multiplies is 32 bits.
// expected-note @+1 {{Parameter declared here.}}
aiex.scratchpad_parameter @n : i64
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%t, MM2S, 0)
  aie.runtime_sequence(%a : memref<1024xbf16>) {
    %task = aiex.dma_configure_task_for @in {
      // expected-error @+1 {{'aie.dma_bd' op size_parameter 'n' must have type i32, got 'i64'.}}
      aie.dma_bd(%a : memref<1024xbf16> offset = 0 len = 1024 sizes = [1, 16, 1, 64] strides = [0, 64, 0, 1]) {bd_id = 0 : i32, size_parameter = @n}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// No third-innermost dimension to size.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%t, MM2S, 0)
  aie.runtime_sequence(%a : memref<1024xbf16>) {
    %task = aiex.dma_configure_task_for @in {
      // expected-error @+1 {{size_parameter patches the third-innermost dimension, so it needs at least three sizes}}
      aie.dma_bd(%a : memref<1024xbf16> offset = 0 len = 1024) {bd_id = 0 : i32, size_parameter = @n}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// A unit of 4 bf16 is 8 bytes: 2 words, which the firmware's mask of the low
// 2 bits of the length would round away.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%t, MM2S, 0)
  aie.runtime_sequence(%a : memref<1024xbf16>) {
    %task = aiex.dma_configure_task_for @in {
      // expected-error @+1 {{a unit must be a multiple of 16 bytes}}
      aie.dma_bd(%a : memref<1024xbf16> offset = 0 len = 64 sizes = [1, 16, 1, 4] strides = [0, 4, 0, 1]) {bd_id = 0 : i32, size_parameter = @n}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// A memtile BD is not rewritten each run.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 1)
  %buf = aie.buffer(%t) : memref<1024xbf16>
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      // expected-error @+1 {{size_parameter is only supported on shim NOC tiles}}
      aie.dma_bd(%buf : memref<1024xbf16> offset = 0 len = 1024 sizes = [1, 16, 1, 64] strides = [0, 64, 0, 1]) {bd_id = 0 : i32, size_parameter = @n}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}

// -----

// A runtime size is written by the dynamic encoder, not a static BD.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%t, MM2S, 0)
  aie.runtime_sequence(%a : memref<1024xbf16>, %k : i64) {
    // expected-error @+1 {{size_parameter needs constant sizes, strides and offsets}}
    aiex.npu.dma_memcpy_nd(%a[0,0,0,0][1,%k,1,64][0,64,0,1]) {id = 0 : i64, metadata = @in, size_parameter = @n} : memref<1024xbf16>
  }
}

// -----

// A d0 of 2048 words does not fit one shim BD, and the parameter patches the
// length of one.
aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%t, MM2S, 0)
  aie.runtime_sequence(%a : memref<65536xi32>) {
    // expected-error @+1 {{has a size_parameter but does not fit one buffer descriptor}}
    aiex.npu.dma_memcpy_nd(%a[0,0,0,0][1,4,2,2048][0,8192,4096,1]) {id = 0 : i64, metadata = @in, size_parameter = @n} : memref<65536xi32>
  }
}

// -----

aiex.scratchpad_parameter @n : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in (%t, MM2S, 0)
  aie.runtime_sequence(%a : memref<65536xi32>) {
    %task = aiex.dma_configure_task_for @in {
      // expected-error @+1 {{has a size_parameter but does not fit one buffer descriptor}}
      aie.dma_bd(%a : memref<65536xi32> offset = 0 len = 16384 sizes = [1, 4, 2, 2048] strides = [0, 8192, 4096, 1]) {bd_id = 0 : i32, size_parameter = @n}
      aie.end
    }
    aiex.dma_start_task(%task)
  }
}
