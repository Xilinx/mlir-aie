//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensures that unranked memref arguments with data layout
// transformations (dimensions/strides/wraps) are correctly handled in the
// dma-tasks-to-npu pass. The length must match the product of the dimension
// sizes.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<*xbf16>) {
      // Dimensions: 8 x 4 x 4 = 128 elements
      // bf16 is 2 bytes, 128 elements = 256 bytes = 64 32-bit words
      // CHECK: aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 2 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 4 : i32, d1_stride = 7 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 31 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK: aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // 3 dimensions: outer size=8, middle size=4, inner size=4
        // Transfer size = 8 * 4 * 4 = 128 bf16 elements
        aie.dma_bd(%arg0 : memref<*xbf16>, 0, 128,
                   [<size=8, stride=64>, <size=4, stride=16>, <size=4, stride=1>]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.dma_start_task(%t1)
      // CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.dma_await_task(%t1)
    }
  }
}
