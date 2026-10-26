//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime-valued dma_bd len on a shim-NOC tile lowers through the dynamic
// BD-word encoder into one npu.blockwrite_values carrying the whole register
// block (bd_id is pinned here, so the address is constant). buffer_length is
// the d0*d1*d2 extent of the (here constant, contiguous) layout, so the runtime
// len is guarded to be non-zero and to agree with it rather than written
// unchecked.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: %[[LEN:.*]] = arith.extui %arg1 : i32 to i64
// CHECK: cf.assert %{{.*}}, "a runtime DMA length must be in [1:4294967295] elements"
// CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[LEN]], %c4096_i64 : i64
// CHECK: cf.assert %[[EQ]], "a runtime DMA length must equal the d0*d1*d2 extent of its dimensions"
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c4096_i32,
// CHECK: aiex.npu.address_patch

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<4096xi32>, %len: i32) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
