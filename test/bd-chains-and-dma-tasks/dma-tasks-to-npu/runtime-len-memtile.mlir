//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime-valued dma_bd len on a MEM TILE lowers through the same dynamic
// BD-word encoder as the shim (see runtime-len.mlir), into one
// npu.blockwrite_values carrying the mem tile's 8-word register block. Two
// things differ from the shim layout, and are what this test pins:
//
//   * buffer_length is 17 bits, so the runtime len is bounded by 131071
//     elements, which the shim's full-width 32-bit field needs no bound for.
//     With dims, len must also equal their d0*d1*d2 extent, so buffer_length
//     is that constant;
//   * the buffer is a local aie.buffer rather than a host argument, so the
//     pointer is written by a maskwrite32 into the 19-bit buffer_offset field
//     instead of by an address patch.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: cf.assert %{{.*}}, "a runtime DMA length must be in [1:131071] elements"
// CHECK: cf.assert %{{.*}}, "a runtime DMA length must equal the d0*d1*d2 extent of its dimensions"
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c4096_i32,
// The local-buffer pointer goes to the mem tile's 19-bit buffer_offset field.
// CHECK: aiex.npu.maskwrite32
// CHECK-NOT: aiex.npu.address_patch
// CHECK-NOT: aiex.npu.writebd

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<4096xi32>

    aie.runtime_sequence(%len: i32) {
      // MM2S channel 0 is even, so the BD id must be below 24
      // (isBdChannelAccessible).
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
