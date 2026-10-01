//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-split-long-repeats %s | FileCheck %s

// A start on a task carried through runtime control flow is split like any
// other: by its own count, or else by the count of the one configure the task
// can come from.

// 301 runs = 256 + 45, read from the configure through the iter_arg. The
// override's 260 runs = 256 + 4 needs no configure.
// CHECK-LABEL: @carried
// CHECK:       scf.for
// CHECK:       aiex.dma_start_task(%[[T:.*]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {repeat_count = 44 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {no_token, repeat_count = 3 : i32}
// CHECK:       scf.yield
// CHECK:       aiex.dma_start_task(%[[R:.*]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[R]]) {repeat_count = 44 : i32}

// The iter_arg can hold either of two configures, so only the override is
// known.
// CHECK-LABEL: @two_configures
// CHECK:       scf.for
// CHECK:       aiex.dma_start_task(%[[P:.*]]){{$}}
// CHECK-NEXT:  aiex.dma_start_task(%[[P]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[P]]) {repeat_count = 3 : i32}
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @carried(%arg0: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {repeat_count = 300 : i32}
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%t = %init) -> (index) {
      aiex.dma_start_task(%t)
      aiex.dma_start_task(%t) {repeat_count = 259 : i32, no_token}
      scf.yield %t : index
    }
    aiex.dma_start_task(%last)
  }
  aie.runtime_sequence @two_configures(%arg0: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {repeat_count = 300 : i32}
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%p = %init) -> (index) {
      aiex.dma_start_task(%p)
      aiex.dma_start_task(%p) {repeat_count = 259 : i32}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {repeat_count = 300 : i32}
      scf.yield %t : index
    }
  }
}
