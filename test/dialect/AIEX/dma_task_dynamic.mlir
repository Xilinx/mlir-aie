//===- dma_task_dynamic.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Round-trip tests for SSA-i64 dynamic operands on aie.dma_bd:
//   * dyn_offset / dyn_len   (Optional<I64>)
//   * dyn_sizes / dyn_strides (Variadic<I64>, all-or-nothing per dim)

// RUN: aie-opt --split-input-file --verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%{{.*}} : memref<128xi32>) dyn_offset(%{{.*}} : i64) {bd_id = 0 : i32}
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<128xi32>) dyn_offset(%M : i64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%{{.*}} : memref<128xi32>) dyn_len(%{{.*}} : i64) {bd_id = 0 : i32}
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<128xi32>) dyn_len(%M : i64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.dma_bd(%{{.*}} : memref<128xi32>) dyn_offset({{.*}}) dyn_len({{.*}}) dyn_sizes({{.*}} : i64, i64, i64, i64) dyn_strides({{.*}} : i64, i64, i64, i64) {bd_id = 0 : i32}
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<128xi32>, %M: i64, %S: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<128xi32>) dyn_offset(%M : i64) dyn_len(%M : i64) dyn_sizes(%M, %S, %M, %S : i64, i64, i64, i64) dyn_strides(%S, %M, %S, %M : i64, i64, i64, i64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
