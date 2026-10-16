//===- reorder_control_packets_gated.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-reorder-control-packets-by-tile | FileCheck %s
// RUN: aie-opt %s -aie-reorder-control-packets-by-tile=by-tile=false | FileCheck %s --check-prefix=PARTITION

// A queue push to a channel aie-check-core-start-gating proved gated (the
// packet's device stamps that channel's aie.dma_start aiex.ctrl_pkt_gated)
// stays with its tile's writes. Pushes to other channels, of another device or
// of no device, and core enables form the enable phase. The addresses are
// 1705524 and 1705532 for memory tile (0, 1)'s MM2S 0 and 1 queues, 2219540 for
// core tile (0, 2)'s MM2S 0 queue, and 2301952 (= 1) for (0, 2)'s core enable.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-NEXT:  address = 1769472 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 11>
// CHECK-NEXT:  address = 1705524 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>
// CHECK-NEXT:  address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 21>
// CHECK-NEXT:  address = 1705532 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// CHECK-NEXT:  address = 1705524 : ui32, aiex.ctrl_pkt_device = @other, aiex.ctrl_pkt_enable_phase, data = array<i32: 2>
// CHECK-NEXT:  address = 2219540 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 2>
// CHECK-NEXT:  address = 2219540 : ui32, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// CHECK-NEXT:  address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>

// Without by-tile only the enable phase moves.
// PARTITION-LABEL: aie.runtime_sequence
// PARTITION-NEXT:  address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 21>
// PARTITION-NEXT:  address = 1769472 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 11>
// PARTITION-NEXT:  address = 1705524 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>
// PARTITION-NEXT:  address = 2219540 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 2>
// PARTITION-NEXT:  address = 1705532 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// PARTITION-NEXT:  address = 1705524 : ui32, aiex.ctrl_pkt_device = @other, aiex.ctrl_pkt_enable_phase, data = array<i32: 2>
// PARTITION-NEXT:  address = 2219540 : ui32, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>
// PARTITION-NEXT:  address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>

module {
  aie.device(npu2) @cfg {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %dma = aie.memtile_dma(%t01) {
      %0 = aie.dma_start(MM2S, 0, ^end, ^next) {aiex.ctrl_pkt_gated}
    ^next:
      %1 = aie.dma_start(MM2S, 1, ^end, ^end)
    ^end:
      aie.end
    }
    %mem = aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^end, ^end)
    ^end:
      aie.end
    }
  }

  aie.device(npu2) @other {
    %t01 = aie.tile(0, 1)
  }

  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet {address = 2355204 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 21>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1769472 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 11>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1705532 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1705524 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1705524 : ui32, aiex.ctrl_pkt_device = @other, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
