//===- dma_task_nodims_linear.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime-offset BD without addressing dims (a contiguous [1, 1, 1, N]
// canonicalizes to this) is a linear transfer of `len` elements. The dynamic
// BD-word encoder must pick linear mode like the static path: words 3 and 4
// (d0 / d1 size and stride) stay constant zero (word 4 keeps only its burst
// bits) instead of an ND encoding with unit wraps.

// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK: aiex.npu.blockwrite_values(%{{.*}}) values %{{.*}}, %{{c0_i32[_0-9]*}}, %{{c0_i32[_0-9]*}}, %{{c0_i32[_0-9]*}}, %c-2147483648_i32, %{{c33554432_i32[_0-9]*}}, %{{.*}}, %{{c33554432_i32[_0-9]*}} : i32, i32, i32, i32, i32, i32, i32, i32
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @seq(%in: memref<4096xi32>, %off: i32) {
      %t = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<4096xi32> offset = %off len = 256) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
