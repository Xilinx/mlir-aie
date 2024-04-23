//===- bad_npu_write_bd_bd.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu) {
    func.func @bad_bd_id(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{BD ID exceeds the maximum ID.}}
      aiex.npu.writebd_shimtile {bd_id = 17 : i32, buffer_length = 32 : i32, buffer_offset = 128 : i32, column = 0 : i32, column_num = 1 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d1_stride = 7 : i32, d1_size = 2 : i32, d2_stride = 15 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_iteration_size(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{Iteration Size exceeds the [0:63] range.}}
      aiex.npu.writebd_shimtile {bd_id = 7 : i32, buffer_length = 32 : i32, buffer_offset = 128 : i32, column = 0 : i32, column_num = 1 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d1_stride = 7 : i32, d1_size = 4 : i32, d2_stride = 15 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 1024 : i32, iteration_size = 128 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_stride(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{D0 Stride exceeds the [0:1M-1] range.}}
      aiex.npu.writebd_shimtile {bd_id = 2 : i32, buffer_length = 32 : i32, buffer_offset = 128 : i32, column = 0 : i32, column_num = 1 : i32, d0_stride = 2097356 : i32, d0_size = 8 : i32, d1_stride = 7 : i32, d1_size = 2 : i32, d2_stride = 15 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_size(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{D1 Size exceeds the [0:1023] range.}}
      aiex.npu.writebd_shimtile {bd_id = 7 : i32, buffer_length = 32 : i32, buffer_offset = 128 : i32, column = 0 : i32, column_num = 1 : i32, d0_stride = 0 : i32, d0_size = 8 : i32, d1_stride = 7 : i32, d1_size = 1024 : i32, d2_stride = 15 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_stride = 0 : i32, iteration_size = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}