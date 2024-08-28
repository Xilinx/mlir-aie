//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// This test ensures that buffer descriptor configurations with data layout transformations are
// properly lowered to the correct NPU instruction sequence instructions.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      // CHECK: aiex.npu.writebd {bd_id = 0 : i32, buffer_length = 4 : i32, buffer_offset = 4 : i32, column = 0 : i32, d0_size = 1 : i32, d0_stride = 0 : i32, d1_size = 2 : i32, d1_stride = 1 : i32, d2_stride = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      // CHECK: aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 4 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8>, 4, 16,
                     [<size=2, stride=4>, <size=2, stride=8>, <size=4, stride=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

