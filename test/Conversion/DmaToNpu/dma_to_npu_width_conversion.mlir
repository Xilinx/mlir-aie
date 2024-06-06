//===- shim_AIE2_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s 2>&1 | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @toMem : memref<65536xbf16>
// CHECK:     func.func @sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
// CHECK:       aiex.npu.writebd_shimtile {bd_id = 0 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 32 : i32, d0_stride = 0 : i32, d1_size = 64 : i32, d1_stride = 127 : i32, d2_stride = 31 : i32, ddr_id = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
// CHECK:       aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147680256 : ui32}
// CHECK:       aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK:       return
// CHECK:       }
// CHECK:       aie.shim_dma_allocation @toMem(S2MM, 0, 0)
// CHECK:     }


module @shimDmaMemcpy{
  aie.device(xcve2302) {
    memref.global "public" @toMem : memref<65536xbf16>
    func.func @sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][4, 4, 64, 64][0, 64, 256]) {id = 0 : i64, metadata = @toMem} : memref<65536xbf16>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}

