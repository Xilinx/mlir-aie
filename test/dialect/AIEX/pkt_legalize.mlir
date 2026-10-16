//===- pkt_legalize.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-legalize-ctrl-packet | FileCheck %s

aie.device(npu2) {
  aie.runtime_sequence @run() {

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

    // The pieces keep the packet's stamps.
    // CHECK: aiex.control_packet {address = 112 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 24, 25, 26, 27>, opcode = 0 : i32, stream_id = 4 : i32}
    // CHECK: aiex.control_packet {address = 128 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 28>, opcode = 0 : i32, stream_id = 4 : i32}
    aiex.control_packet {address = 112 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 24, 25, 26, 27, 28>, opcode = 0 : i32, stream_id = 4 : i32}

  }
}
