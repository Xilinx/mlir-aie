//===- dma_to_npu_core_tile.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// Verify WriteBdOp packing for a core tile
// CHECK-LABEL: module
// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<6xi32> = dense<[1180485, 1430061056, 9093684, 266847009, 22249971, 1465218380]>
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 2216128 : ui32} : memref<6xi32>
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence() {
      aiex.npu.writebd {
        bd_id = 6 : i32,
        buffer_length = 837 : i32,
        buffer_offset = 291 : i32,
        enable_packet = 1 : i32,
        out_of_order_id = 21 : i32,
        packet_id = 7 : i32,
        packet_type = 5 : i32,
        column = 0 : i32,
        row = 2 : i32,
        d0_stride = 564 : i32,
        d0_size = 62 : i32,
        d0_zero_after = 0 : i32,
        d0_zero_before = 0 : i32,
        d1_stride = 1110 : i32,
        d1_size = 127 : i32,
        d1_zero_after = 0 : i32,
        d1_zero_before = 0 : i32,
        d2_size = 0 : i32,
        d2_stride = 801 : i32,
        d2_zero_after = 0 : i32,
        d2_zero_before = 0 : i32,
        ddr_id = 0 : i32,
        iteration_current = 42 : i32,
        iteration_stride = 499 : i32,
        iteration_size = 28 : i32,
        lock_acq_enable = 1 : i32,
        lock_acq_id = 12 : i32,
        lock_acq_val = 42 : i32,
        lock_rel_id = 11 : i32,
        lock_rel_val = 85 : i32,
        next_bd = 10 : i32,
        use_next_bd = 1 : i32,
        valid_bd = 1 : i32,
        burst_length = 0 : i32
      }
    }
  }
}
