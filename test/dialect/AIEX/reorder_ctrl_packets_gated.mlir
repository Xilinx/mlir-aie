//===- reorder_ctrl_packets_gated.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-schedule-ctrl-packet-activation | FileCheck %s --check-prefixes=CHECK,CONTEXT
// RUN: aie-opt %s --split-input-file -aie-schedule-ctrl-packet-activation=gated-enable=true | FileCheck %s --check-prefixes=GATED,CONTEXT

// A queue push to a channel aie-annotate-ctrl-packet-start-gating proved gated
// (the selected device marks that channel's aie.dma_start
// aiex.started_by) stays with its tile's writes. Pushes to other
// channels and core enables form the enable phase, after a barrier. The
// addresses are 1705524 and 1705532 for memory tile (0, 1)'s MM2S 0 and 1
// queues, 2219540 for core tile (0, 2)'s MM2S 0 queue, 2301952 (= 1) for
// (0, 2)'s core enable, and 35909636 for tile (1, 2)'s switch. The marker for
// @other starts a separate run.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-NEXT:  aiex.set_device @cfg
// CHECK-NEXT:    address = 1769472 : ui32, data = array<i32: 11>
// CHECK-NEXT:    address = 1705524 : ui32, data = array<i32: 1>
// CHECK-NEXT:    address = 2355204 : ui32, data = array<i32: 21>
// CHECK-NEXT:    address = 35909636 : ui32, data = array<i32: 31>
// CHECK-NEXT:  aiex.control_packet_barrier
// CHECK-NEXT:    address = 1705532 : ui32, data = array<i32: 1>
// CHECK-NEXT:    address = 2219540 : ui32, data = array<i32: 2>
// CHECK-NEXT:    address = 2301952 : ui32, data = array<i32: 1>
// CHECK-NEXT:  aiex.set_device @other
// CHECK-NEXT:    address = 1705524 : ui32, data = array<i32: 2>
// CHECK-NEXT:  address = 2219540 : ui32, data = array<i32: 1>

// With gated-enable, the enable of core (0, 2), which @cfg marks gated, goes
// last among column 0's packets.
// GATED-LABEL: aie.runtime_sequence
// GATED-NEXT:  aiex.set_device @cfg
// GATED-NEXT:    address = 1769472 : ui32, data = array<i32: 11>
// GATED-NEXT:    address = 1705524 : ui32, data = array<i32: 1>
// GATED-NEXT:    address = 2355204 : ui32, data = array<i32: 21>
// GATED-NEXT:    address = 2301952 : ui32, data = array<i32: 1>
// GATED-NEXT:    address = 35909636 : ui32, data = array<i32: 31>
// GATED-NEXT:  aiex.control_packet_barrier
// GATED-NEXT:    address = 1705532 : ui32, data = array<i32: 1>
// GATED-NEXT:    address = 2219540 : ui32, data = array<i32: 2>

module {
  aie.device(npu2) @cfg {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %dma = aie.memtile_dma(%t01) {
      %0 = aie.dma_start(MM2S, 0, ^end, ^next) {aiex.started_by = []}
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
    %core = aie.core(%t02) {
      aie.end
    } {aiex.started_by = []}
  }

  aie.device(npu2) @other {
    %t01 = aie.tile(0, 1)
  }

  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.set_device @cfg
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 21>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1769472 : ui32, data = array<i32: 11>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1705532 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35909636 : ui32, data = array<i32: 31>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_device @other
      aiex.control_packet {address = 1705524 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// CONTEXT-LABEL: aie.runtime_sequence @nested_context
// CONTEXT:      aiex.set_device @cfg
// CONTEXT-NEXT: scf.for
// CONTEXT-NEXT:   aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>
// CONTEXT-NEXT:   aiex.control_packet {address = 1769472 : ui32, data = array<i32: 11>
// CONTEXT-NEXT:   aiex.set_device @other
// CONTEXT-NEXT:   aiex.control_packet {address = 1769472 : ui32, data = array<i32: 12>
// CONTEXT-NEXT:   aiex.control_packet_barrier
// CONTEXT-NEXT:   aiex.control_packet {address = 1705524 : ui32, data = array<i32: 2>
// CONTEXT-NEXT: {{^ *}}}
// CONTEXT-NEXT: aiex.control_packet {address = 1705524 : ui32, data = array<i32: 3>
// CONTEXT-NEXT: aiex.control_packet {address = 1769472 : ui32, data = array<i32: 13>
// CONTEXT-NEXT: aiex.set_device @cfg
// CONTEXT-NEXT: aiex.control_packet {address = 35909636 : ui32, data = array<i32: 31>
// CONTEXT-NEXT: aiex.set_device @cfg
// CONTEXT-NEXT: aiex.control_packet {address = 1769472 : ui32, data = array<i32: 14>
// CONTEXT-NEXT: {{^ *}}}
module {
  aie.device(npu2) @cfg {
    %memory = aie.tile(0, 1)
    aie.memtile_dma(%memory) {
      aie.dma_start(MM2S, 0, ^end, ^end) {aiex.started_by = []}
    ^end:
      aie.end
    }
  }
  aie.device(npu2) @other {
    %memory = aie.tile(0, 1)
  }
  aie.device(npu2) {
    aie.runtime_sequence @nested_context() {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      aiex.set_device @cfg
      scf.for %iteration = %zero to %one step %one {
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1769472 : ui32, data = array<i32: 11>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.set_device @other
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1769472 : ui32, data = array<i32: 12>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet {address = 1705524 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1769472 : ui32, data = array<i32: 13>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_device @cfg
      aiex.control_packet {address = 35909636 : ui32, data = array<i32: 31>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_device @cfg
      aiex.control_packet {address = 1769472 : ui32, data = array<i32: 14>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
