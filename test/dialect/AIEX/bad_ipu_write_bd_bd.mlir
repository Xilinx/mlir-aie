//===- bad_ipu_write_bd_bd.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module {
  AIE.device(ipu) {
    func.func @sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{BD ID exceeds the maximum ID.}}
      AIEX.ipu.writebd_shimtile {bd_id = 17 : i32, buffer_length = 32 : i32, buffer_offset = 128 : i32, column = 0 : i32, column_num = 1 : i32, d0_stepsize = 0 : i32, d0_wrap = 8 : i32, d1_stepsize = 7 : i32, d1_wrap = 2 : i32, d2_stepsize = 15 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stepsize = 0 : i32, iteration_wrap = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      return
    }
    AIE.shimDMAAllocation @of_fromMem (MM2S, 0, 0)
  }
}