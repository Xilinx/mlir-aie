//===- pkt_legalize.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-legalize-ctrl-packet | FileCheck %s

aie.device(npu2) {
  aiex.runtime_sequence @run() {

    // CHECK: aiex.control_packet {address = 8 : ui32, data = array<i32: 1, 2, 3, 4>, opcode = 0 : i32, stream_id = 1 : i32}
    // CHECK: aiex.control_packet {address = 24 : ui32, data = array<i32: 5, 6>, opcode = 0 : i32, stream_id = 1 : i32}
    aiex.control_packet {address = 8 : ui32, data = array<i32: 1, 2, 3, 4, 5, 6>, opcode = 0 : i32, stream_id = 1 : i32}

    // CHECK: aiex.control_packet {address = 30 : ui32, data = array<i32: 7, 8, 9, 10>, opcode = 0 : i32, stream_id = 2 : i32}
    aiex.control_packet {address = 30 : ui32, data = array<i32: 7, 8, 9, 10>, opcode = 0 : i32, stream_id = 2 : i32}

    // CHECK: aiex.control_packet {address = 40 : ui32, data = array<i32: 11, 12, 13, 14>, opcode = 0 : i32, stream_id = 3 : i32}
    // CHECK: aiex.control_packet {address = 56 : ui32, data = array<i32: 15, 16, 17, 18>, opcode = 0 : i32, stream_id = 3 : i32}
    // CHECK: aiex.control_packet {address = 72 : ui32, data = array<i32: 19, 20, 21, 22>, opcode = 0 : i32, stream_id = 3 : i32}
    // CHECK: aiex.control_packet {address = 88 : ui32, data = array<i32: 23>, opcode = 0 : i32, stream_id = 3 : i32}
    aiex.control_packet {address = 40 : ui32, data = array<i32: 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>, opcode = 0 : i32, stream_id = 3 : i32}

  }
}