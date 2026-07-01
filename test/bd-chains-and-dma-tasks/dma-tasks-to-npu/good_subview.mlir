//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Test that memref.subview is correctly traced through when configuring DMA tasks

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<1024xi16>) {
      // Create a subview at offset 64 elements (128 bytes for i16)
      // CHECK: aiex.npu.writebd
      // CHECK-DAG: %[[AP128:.*]] = arith.constant 128 : i32
      // CHECK: aiex.npu.address_patch(%[[AP128]] : i32) {addr = {{.*}}, arg_idx = 0 : i32}
      %subview = memref.subview %arg0[64] [128] [1] : memref<1024xi16> to memref<128xi16, strided<[1], offset: 64>>
      %reinterpret = memref.reinterpret_cast %subview to offset: [0], sizes: [128], strides: [1] : memref<128xi16, strided<[1], offset: 64>> to memref<128xi16>
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%reinterpret : memref<128xi16>, 0, 128) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      
      // CHECK-DAG: %[[PQBD:.*]] = arith.constant 7 : i32
      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[PQBD]] repeat %{{.*}} {issue_token = true} : i32, i32
      aiex.dma_start_task(%t1)
      // sync operands: column=0, row=0, direction=1, channel=0, column_num=1, row_num=1
      // CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t1)
    }
  }
}
