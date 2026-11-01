//===- good-iteration-runtime-dims.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A runtime-valued dimension sends a static-bd_id BD through the dynamic
// BD-word encoder, which still packs #aie.bd_iteration into shim word 6:
// wrap 3 at bit 20, stride 15 (0x0030000F).

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// CHECK: %[[W6:.*]] = arith.constant 3145743 : i32
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[W6]], %{{.*}} : i32, i32, i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>, %n: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [%n, 8, 4] strides = [512, 4, 1]) { bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%t)
    }
  }
}
