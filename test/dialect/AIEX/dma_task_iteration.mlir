//===- dma_task_iteration.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu %s | FileCheck %s

// On the runtime-sequence path the #aie.bd_iteration attribute fills the
// iteration register, as an outermost sizes/strides dimension would, and also
// sets its starting step.

// CHECK-LABEL: aie.runtime_sequence @shim
// CHECK: aiex.npu.writebd {axcache = 2 : i32, bd_id = 0 : i32, {{.*}}d0_size = 8 : i32, {{.*}}d1_size = 4 : i32, d1_stride = 15 : i32, {{.*}}d2_stride = 7 : i32, {{.*}}iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 127 : i32
// CHECK: aiex.npu.writebd {axcache = 2 : i32, bd_id = 1 : i32, {{.*}}d0_size = 8 : i32, {{.*}}d1_size = 4 : i32, d1_stride = 15 : i32, {{.*}}d2_stride = 7 : i32, {{.*}}iteration_current = 0 : i32, iteration_size = 1 : i32, iteration_stride = 127 : i32

// A BD without dims is one contiguous dimension under the iteration.
// CHECK-LABEL: aie.runtime_sequence @memtile
// CHECK: aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 64 : i32, {{.*}}iteration_current = 1 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32
// CHECK: aiex.npu.writebd {bd_id = 1 : i32, buffer_length = 64 : i32, {{.*}}iteration_current = 0 : i32, iteration_size = 3 : i32, iteration_stride = 63 : i32

// The dynamic encoder packs the same fields: word 6 holds iteration_current 1,
// iteration_size 3 and iteration_stride 63 (0x86003F).
// CHECK-LABEL: aie.runtime_sequence @memtile_runtime_offset
// CHECK: %[[W6:.*]] = arith.constant 8781887 : i32
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[W6]], %{{.*}} : i32, i32, i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "buf"} : memref<256xi32>

    aie.runtime_sequence @shim(%arg0: memref<256xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64 sizes = [2, 4, 8] strides = [8, 16, 1]) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 2, stride = 128, current = 0>}
        aie.end
      }
      aiex.dma_start_task(%t)
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64 sizes = [2, 2, 4, 8] strides = [128, 8, 16, 1]) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
    }

    aie.runtime_sequence @memtile() {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 64, current = 1>}
        aie.end
      }
      aiex.dma_start_task(%t)
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 64 sizes = [4, 1, 1, 64] strides = [64, 0, 0, 1]) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
    }

    aie.runtime_sequence @memtile_runtime_offset(%off: i32) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = %off len = 64) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 64, current = 1>}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
