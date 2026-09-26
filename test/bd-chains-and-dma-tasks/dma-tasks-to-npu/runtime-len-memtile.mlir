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
//   * buffer_length is 17 bits and shares word 0 with the packet header, so
//     the runtime length is masked to 0x1ffff and OR'd into a constant
//     template -- and guarded against 131071, which the shim's full-width
//     32-bit field needs no guard for;
//   * the buffer is a local aie.buffer rather than a host argument, so the
//     pointer is written by a maskwrite32 into the 19-bit buffer_offset field
//     instead of by an address patch.

// CHECK-LABEL: aie.runtime_sequence
// buffer_length = len * elemWidth / addressGranularity, from the runtime %len:
// CHECK: aiex.npu.assert_bd_field(%arg0) {max = 131071 : i32}
// CHECK: %[[DIV:.*]] = arith.divui %arg0, %{{.*}}
// CHECK: %[[BLEN:.*]] = arith.muli %[[DIV]], %{{.*}}
// The 17-bit buffer_length guard, emitted only because the field is narrow:
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 131071 : i32}
// CHECK: aiex.npu.blockwrite_values
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
