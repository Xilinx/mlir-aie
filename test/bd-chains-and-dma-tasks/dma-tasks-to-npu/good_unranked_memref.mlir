//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensures that unranked memref arguments are correctly handled
// in the dma-tasks-to-npu pass. Unranked memrefs require explicit length
// specification since the buffer size cannot be inferred from the type.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<*xbf16>) {
      // CHECK: aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK-DAG: %[[AP0:.*]] = arith.constant 0 : i32
      // CHECK: aiex.npu.address_patch(%[[AP0]] : i32) {addr = 118788 : ui32, arg_idx = 0 : i32}
      %c0_i32 = arith.constant 0 : i32
      %c128_i32 = arith.constant 128 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // bf16 is 2 bytes, 128 elements = 256 bytes = 64 32-bit words
        aie.dma_bd(%arg0 : memref<*xbf16> offset = %c0_i32 len = %c128_i32) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %{{.*}} {issue_token = true} : i32, i32
      aiex.dma_start_task(%t1)
      // sync operands: column=0, row=0, direction=1, channel=0, column_num=1, row_num=1
      // CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t1)
    }
  }
}
