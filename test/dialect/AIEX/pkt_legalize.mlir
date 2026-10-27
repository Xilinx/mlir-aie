//===- pkt_legalize.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-legalize-ctrl-packet | FileCheck %s

// A packet carries at most four data words, which have to stay in one 16-byte
// line, because on the device the words of a packet that crosses a line
// boundary wrap to the start of its line.

aie.device(npu2) {
  aie.runtime_sequence @run() {

    // CHECK: aiex.control_packet {address = 8 : ui32, data = array<i32: 1, 2>, opcode = 0 : i32, stream_id = 1 : i32}
    // CHECK: aiex.control_packet {address = 16 : ui32, data = array<i32: 3, 4, 5, 6>, opcode = 0 : i32, stream_id = 1 : i32}
    aiex.control_packet {address = 8 : ui32, data = array<i32: 1, 2, 3, 4, 5, 6>, opcode = 0 : i32, stream_id = 1 : i32}

    // Four words or fewer are split too when they cross a line.
    // CHECK: aiex.control_packet {address = 28 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 2 : i32}
    // CHECK: aiex.control_packet {address = 32 : ui32, data = array<i32: 8, 9, 10>, opcode = 0 : i32, stream_id = 2 : i32}
    aiex.control_packet {address = 28 : ui32, data = array<i32: 7, 8, 9, 10>, opcode = 0 : i32, stream_id = 2 : i32}

    // CHECK: aiex.control_packet {address = 40 : ui32, data = array<i32: 11, 12>, opcode = 0 : i32, stream_id = 3 : i32}
    // CHECK: aiex.control_packet {address = 48 : ui32, data = array<i32: 13, 14, 15, 16>, opcode = 0 : i32, stream_id = 3 : i32}
    // CHECK: aiex.control_packet {address = 64 : ui32, data = array<i32: 17, 18, 19, 20>, opcode = 0 : i32, stream_id = 3 : i32}
    // CHECK: aiex.control_packet {address = 80 : ui32, data = array<i32: 21, 22, 23>, opcode = 0 : i32, stream_id = 3 : i32}
    aiex.control_packet {address = 40 : ui32, data = array<i32: 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>, opcode = 0 : i32, stream_id = 3 : i32}

    // An aligned packet of four words or fewer is unchanged.
    // CHECK: aiex.control_packet {address = 96 : ui32, data = array<i32: 24, 25, 26, 27>, opcode = 0 : i32, stream_id = 4 : i32}
    // CHECK-NOT: aiex.control_packet
    aiex.control_packet {address = 96 : ui32, data = array<i32: 24, 25, 26, 27>, opcode = 0 : i32, stream_id = 4 : i32}

  }
}
