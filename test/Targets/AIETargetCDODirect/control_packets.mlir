//===- control_packets.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-ctrlpkt-to-bin -aie-output-binary=false %s |& FileCheck %s

// CHECK: 8000001B
// CHECK: 0001F000
// CHECK: 00000002
// CHECK: 8000001B
// CHECK: 09B1F020
// CHECK: 00000003
// CHECK: 00000004
// CHECK: 00000005
// CHECK: 00000006
// CHECK: 8000001B
// CHECK: 02700400
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.runtime_sequence() {
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 127008 : ui32, data = array<i32: 3, 4, 5, 6>, opcode = 2 : i32, stream_id = 9 : i32}
      aiex.control_packet {address = 1024 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
    }
  }
}
