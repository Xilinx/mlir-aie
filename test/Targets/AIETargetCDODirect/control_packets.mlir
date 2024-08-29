//===- control_packets.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-ctrlpkt-to-bin %s |& FileCheck %s

// CHECK: 0001F000
// CHECK: 00000002
// CHECK: 09B1F020
// CHECK: 00000003
// CHECK: 00000004
// CHECK: 00000005
// CHECK: 00000006
// CHECK: 02700400
module {
  aie.device(npu1) {
    aiex.runtime_sequence() {
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 127008 : ui32, data = array<i32: 3, 4, 5, 6>, opcode = 2 : i32, stream_id = 9 : i32}
      aiex.control_packet {address = 1024 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
    }
  }
}
