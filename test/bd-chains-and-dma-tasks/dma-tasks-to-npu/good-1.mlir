//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensures buffer descriptor configurations, as well as `aiex.dma_start_task`,
// `aiex.dma_await_task` operations, issued from within the runtime sequence,
// are lowered to the correct NPU instruction sequence instructions.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)

    aie.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<10xi32>) {
      // CHECK: aiex.npu.writebd {bd_id = 7 : i32, buffer_length = 4 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 119012 : ui32, arg_idx = 0 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
      // CHECK: aiex.npu.writebd {bd_id = 8 : i32, buffer_length = 10 : i32, buffer_offset = 0 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK: aiex.npu.address_patch(%{{.*}} : i32) {addr = 67227908 : ui32, arg_idx = 1 : i32}
      %t2 = aiex.dma_configure_task(%tile_2_0, S2MM, 1) {
        aie.dma_bd(%arg1 : memref<10xi32>, 0, 10) {bd_id = 8 : i32}
        aie.end
      } {repeat_count = 2 : i32, issue_token = true}

      // CHECK-DAG: %[[T1BD:.*]] = arith.constant 7 : i32
      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[T1BD]] repeat %{{.*}} {issue_token = true} : i32, i32
      aiex.dma_start_task(%t1)
      // CHECK-DAG: %[[T2BD:.*]] = arith.constant 8 : i32
      // CHECK-DAG: %[[T2RC:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.push_queue(2, 0, S2MM : 1) bd_id %[[T2BD]] repeat %[[T2RC]] {issue_token = true} : i32, i32
      aiex.dma_start_task(%t2)
      // sync operands: column=0, row=0, direction=1, channel=0, column_num=1, row_num=1
      // CHECK: %[[S1RN:.*]] = arith.constant 1 : i32
      // CHECK: %[[S1CN:.*]] = arith.constant 1 : i32
      // CHECK: %[[S1CH:.*]] = arith.constant 0 : i32
      // CHECK: %[[S1DIR:.*]] = arith.constant 1 : i32
      // CHECK: %[[S1ROW:.*]] = arith.constant 0 : i32
      // CHECK: %[[S1COL:.*]] = arith.constant 0 : i32
      // CHECK: aiex.npu.sync(%[[S1COL]], %[[S1ROW]], %[[S1DIR]], %[[S1CH]], %[[S1CN]], %[[S1RN]]) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t1)
      // sync operands: column=2, row=0, direction=0, channel=1, column_num=1, row_num=1
      // CHECK: %[[S2RN:.*]] = arith.constant 1 : i32
      // CHECK: %[[S2CN:.*]] = arith.constant 1 : i32
      // CHECK: %[[S2CH:.*]] = arith.constant 1 : i32
      // CHECK: %[[S2DIR:.*]] = arith.constant 0 : i32
      // CHECK: %[[S2ROW:.*]] = arith.constant 0 : i32
      // CHECK: %[[S2COL:.*]] = arith.constant 2 : i32
      // CHECK: aiex.npu.sync(%[[S2COL]], %[[S2ROW]], %[[S2DIR]], %[[S2CH]], %[[S2CN]], %[[S2RN]]) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t2)
    }
  }
}