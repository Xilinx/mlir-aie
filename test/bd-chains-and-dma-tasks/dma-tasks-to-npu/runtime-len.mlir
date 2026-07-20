//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime-valued dma_bd len on a shim-NOC tile lowers through the dynamic
// BD-word encoder: a zero-template writebd (folded to one blockwrite later)
// plus per-word write32 overrides. buffer_length carries the runtime len as
// len * elemWidth / addressGranularity; the size/stride words carry the
// (here constant) ND layout.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: aiex.npu.writebd {{.*}}buffer_length = 0{{.*}}valid_bd = 1
// buffer_length override derived from the runtime %len operand:
// CHECK: %[[MUL:.*]] = arith.muli %arg1, %{{.*}}
// CHECK: arith.divui %[[MUL]], %{{.*}}
// CHECK: aiex.npu.write32
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
