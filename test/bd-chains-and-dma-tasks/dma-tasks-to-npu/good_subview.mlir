//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Test that memref.subview is correctly traced through when configuring DMA tasks

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<1024xi16>) {
      // Create a subview at offset 64 elements (128 bytes for i16)
      // CHECK: aiex.npu.writebd
      // CHECK: aiex.npu.address_patch {addr = {{.*}}, arg_idx = 0 : i32, arg_plus = 128 : i32}
      %subview = memref.subview %arg0[64] [128] [1] : memref<1024xi16> to memref<128xi16, strided<[1], offset: 64>>
      %reinterpret = memref.reinterpret_cast %subview to offset: [0], sizes: [128], strides: [1] : memref<128xi16, strided<[1], offset: 64>> to memref<128xi16>
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%reinterpret : memref<128xi16>, 0, 128) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      
      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 7 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.dma_start_task(%t1)
      // CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.dma_await_task(%t1)
    }
  }
}
