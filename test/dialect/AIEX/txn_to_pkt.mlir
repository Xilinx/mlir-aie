//===- txn_to_pkt.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-txn-to-ctrl-packet | FileCheck %s
// CHECK: aiex.control_packet {address = 65432 : ui32, data = array<i32: 123, 456, 789, 43981, -2147483648, -1, -2, -3>

module {
  aie.device(npu2_1col) {
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[123, 456, 789, 0xabcd, 0x80000000, -1, -2, -3]>
    aiex.runtime_sequence @run() {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 65432 : ui32} : memref<8xi32>
    }
  }
}
