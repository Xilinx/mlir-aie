//===- good-iteration-attribute.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A compile-time-constant BD's #aie.bd_iteration attribute lowers to the raw
// npu.writebd iteration_current/size/stride fields: true element values (size
// = 4, stride = 16) become the off-by-one/word-scaled register form (3, 15),
// same conversion AIERT.cpp applies on the structural path; current (2) is
// unbiased. The second task combines it with a real 3-dimensional access
// pattern, showing the attribute does not clobber that encoding (it claims
// the would-be-hoisted 4th slot instead, per verifyTaskBDDimensions).

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<8192xi32>) {
      // CHECK: aiex.npu.writebd {axcache = 2 : i32, bd_id = 5 : i32, buffer_length = 64 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 2 : i32, iteration_size = 3 : i32, iteration_stride = 15 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)

      // CHECK: aiex.npu.writebd {axcache = 2 : i32, bd_id = 6 : i32, buffer_length = 32 : i32, buffer_offset = 0 : i32, column = 0 : i32, d0_size = 1 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 8 : i32, d1_stride = 31 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 511 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 2 : i32, iteration_size = 3 : i32, iteration_stride = 15 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg1 : memref<8192xi32> offset = 0 len = 32 sizes = [4, 8, 1] strides = [512, 32, 1]) {bd_id = 6 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
    }
  }
}
