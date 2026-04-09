//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// Tests that >16 tasks on one shim tile succeed when dma_free_task
// recycles BD IDs between batches. Two host buffers alternate on
// the same MM2S channel (simulates RoPE LUT + V interleaving).
//
// Batch 1: 10 tasks → BD IDs 0-9.
// Free all → IDs 0-9 available.
// Batch 2: 10 tasks → reuses BD IDs 0-9.
// Total: 20 tasks on one tile (exceeds 16 limit without reuse).

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%buf_a: memref<4096xbf16>, %buf_b: memref<4096xbf16>) {

      // ===== Batch 1: 10 tasks, BD IDs 0-9 =====

      // CHECK: aie.dma_bd(%arg0 {{.*}} offset = 0 {{.*}} {bd_id = 0 : i32}
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 0, 4096)
        aie.end
      }
      aiex.dma_start_task(%t0)

      // CHECK: aie.dma_bd(%arg1 {{.*}} offset = 0 {{.*}} {bd_id = 1 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 0, 4096)
        aie.end
      }
      aiex.dma_start_task(%t1)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 2 : i32}
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 4096, 4096)
        aie.end
      }
      aiex.dma_start_task(%t2)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 3 : i32}
      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 4096, 4096)
        aie.end
      }
      aiex.dma_start_task(%t3)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 4 : i32}
      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 8192, 4096)
        aie.end
      }
      aiex.dma_start_task(%t4)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 5 : i32}
      %t5 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 8192, 4096)
        aie.end
      }
      aiex.dma_start_task(%t5)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 6 : i32}
      %t6 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 12288, 4096)
        aie.end
      }
      aiex.dma_start_task(%t6)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 7 : i32}
      %t7 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 12288, 4096)
        aie.end
      }
      aiex.dma_start_task(%t7)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 8 : i32}
      %t8 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 16384, 4096)
        aie.end
      }
      aiex.dma_start_task(%t8)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 9 : i32}
      %t9 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 16384, 4096)
        aie.end
      }
      aiex.dma_start_task(%t9)

      // Await last task (guarantees all prior sequential tasks completed),
      // then free each task's BDs individually to recycle IDs.
      aiex.dma_await_task(%t9)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
      aiex.dma_free_task(%t4)
      aiex.dma_free_task(%t5)
      aiex.dma_free_task(%t6)
      aiex.dma_free_task(%t7)
      aiex.dma_free_task(%t8)

      // ===== Batch 2: 10 more tasks, reuses BD IDs 0-9 =====

      // CHECK: aie.dma_bd(%arg0 {{.*}} offset = 20480 {{.*}} {bd_id = 0 : i32}
      %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 20480, 4096)
        aie.end
      }
      aiex.dma_start_task(%t10)

      // CHECK: aie.dma_bd(%arg1 {{.*}} offset = 20480 {{.*}} {bd_id = 1 : i32}
      %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 20480, 4096)
        aie.end
      }
      aiex.dma_start_task(%t11)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 2 : i32}
      %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 24576, 4096)
        aie.end
      }
      aiex.dma_start_task(%t12)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 3 : i32}
      %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 24576, 4096)
        aie.end
      }
      aiex.dma_start_task(%t13)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 4 : i32}
      %t14 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 28672, 4096)
        aie.end
      }
      aiex.dma_start_task(%t14)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 5 : i32}
      %t15 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 28672, 4096)
        aie.end
      }
      aiex.dma_start_task(%t15)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 6 : i32}
      %t16 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 32768, 4096)
        aie.end
      }
      aiex.dma_start_task(%t16)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 7 : i32}
      %t17 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 32768, 4096)
        aie.end
      }
      aiex.dma_start_task(%t17)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 8 : i32}
      %t18 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_a : memref<4096xbf16>, 36864, 4096)
        aie.end
      }
      aiex.dma_start_task(%t18)

      // CHECK: aie.dma_bd({{.*}} {bd_id = 9 : i32}
      %t19 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%buf_b : memref<4096xbf16>, 36864, 4096)
        aie.end
      }
      aiex.dma_start_task(%t19)

      aiex.dma_await_task(%t19)
    }
  }
}
