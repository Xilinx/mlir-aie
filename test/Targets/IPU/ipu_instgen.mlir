//===- ipu_instgen.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-ipu-instgen %s | FileCheck %s
module {
  aie.device(ipu) {
    func.func @test0(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {

      // look for the prolog.
      // CHECK:      00000011
      // CHECK-NEXT: 01000405
      // CHECK-NEXT: 01000100
      // CHECK-NEXT: 0B590100
      // CHECK-NEXT: 000055FF
      // CHECK-NEXT: 00000001
      // CHECK-NEXT: 00000010
      // CHECK-NEXT: 314E5A5F
      // CHECK-NEXT: 635F5F31
      // CHECK-NEXT: 676E696C
      // CHECK-NEXT: 39354E5F
      // CHECK-NEXT: 6E693131
      // CHECK-NEXT: 5F727473
      // CHECK-NEXT: 64726F77
      // CHECK-NEXT: 00004573
      // CHECK-NEXT: 07BD9630
      // CHECK-NEXT: 000055FF

      %c16_i64 = arith.constant 16 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i64 = arith.constant 0 : i64
      %c64_i64 = arith.constant 64 : i64
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      // CHECK: 060304A6
      // CHECK: 00000000
      // CHECK: 00000001
      // CHECK: 00000002
      // CHECK: 00000000
      // CHECK: 00600005
      // CHECK: 80800007
      // CHECK: 00000009
      // CHECK: 2CD0000C
      // CHECK: 2E107041
      aiex.ipu.writebd_shimtile { bd_id = 6 : i32,
                                  buffer_length = 1 : i32,
                                  buffer_offset = 2 : i32,
                                  enable_packet = 0 : i32,
                                  out_of_order_id = 0 : i32,
                                  packet_id = 0 : i32,
                                  packet_type = 0 : i32,
                                  column = 3 : i32,
                                  column_num = 4 : i32,
                                  d0_stride = 5 : i32,
                                  d0_size = 6 : i32,
                                  d1_stride = 7 : i32,
                                  d1_size = 8 : i32,
                                  d2_stride = 9 : i32,
                                  ddr_id = 10 : i32,
                                  iteration_current = 11 : i32,
                                  iteration_stride = 12 : i32,
                                  iteration_size = 13 : i32,
                                  lock_acq_enable = 1 : i32,
                                  lock_acq_id = 1 : i32,
                                  lock_acq_val = 2 : i32,
                                  lock_rel_id = 3 : i32,
                                  lock_rel_val = 4 : i32,
                                  next_bd = 5 : i32,
                                  use_next_bd = 1 : i32,
                                  valid_bd = 1 : i32}
      // CHECK: 02030400
      // CHECK: ABC00DEF
      // CHECK: 00000042
      aiex.ipu.write32 { column = 3 : i32, row = 4 : i32, address = 0xabc00def : ui32, value = 0x42 : ui32 }
      // CHECK: 03030401
      // CHECK: 05010200
      aiex.ipu.sync { column = 3 : i32, row = 4 : i32, direction = 1 : i32, channel = 5 : i32, column_num = 1 : i32, row_num = 2 : i32 }
      return
    }
  }
}
