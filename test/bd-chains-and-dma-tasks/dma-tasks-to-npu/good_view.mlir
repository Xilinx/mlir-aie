//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Test that memref.view is correctly traced through when configuring DMA tasks.
// This is the pattern the fused whole-model (--get-full-elf) lowering emits:
// every DMA buffer is a typed memref.view slice of one flat byte-arena runtime
// sequence input argument at a constant byte offset. Without tracing view, the
// buffer fails to resolve to its block argument and the BD cannot be lowered.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<1024xi8>) {
      // View at byte offset 128 -> address patch arg_plus = 128.
      // CHECK: aiex.npu.writebd
      // CHECK-DAG: %[[AP128:.*]] = arith.constant 128 : i32
      // CHECK: aiex.npu.address_patch(%[[AP128]] : i32) {addr = {{.*}}, arg_idx = 0 : i32}
      %c128 = arith.constant 128 : index
      %view = memref.view %arg0[%c128][] : memref<1024xi8> to memref<128xi16>
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%view : memref<128xi16> offset = 0 len = 128) {bd_id = 7 : i32}
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
