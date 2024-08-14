//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    //%tile_2_0 = aie.tile(2, 0)

    aie.shim_dma_allocation @alloc0 (MM2S, 0, 0)
    aie.shim_dma_allocation @alloc1 (S2MM, 1, 2)

    aiex.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<10xi32>) {
      // CHECK: aiex.npu.writebd {bd_id = 7 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK: aiex.npu.address_patch {addr = 119012 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      %t1 = aiex.dma_configure_task_for @alloc0 {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      // CHECK: aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 10 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK: aiex.npu.address_patch {addr = 67227908 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32} 
      %t2 = aiex.dma_configure_task_for @alloc1 {
        aie.dma_bd(%arg1 : memref<10xi32>, 0, 10) {bd_id = 8 : i32}
        aie.end
      } {repeat_count = 2 : i32, issue_token = true}

      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 7 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.dma_start_task(%t1)
      // CHECK: aiex.npu.push_queue(2, 0, S2MM : 1) {bd_id = 8 : i32, issue_token = true, repeat_count = 2 : i32}
      aiex.dma_start_task(%t2)
      // CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.dma_await_task(%t1)
      // CHECK: aiex.npu.sync {channel = 1 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.dma_await_task(%t2)
    }
  }
}

