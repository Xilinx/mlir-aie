//===- reorder_ctrl_packets_stall.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-reorder-ctrl-packets-by-tile -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,SERIAL
// RUN: aie-opt %s --split-input-file -aie-reorder-ctrl-packets-by-tile=parallel-shims=true -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,PARALLEL
// RUN: aie-opt %s --split-input-file -aie-reorder-ctrl-packets-by-tile -verify-diagnostics | aie-opt --split-input-file -aie-check-ctrl-packet-delivery
// RUN: aie-opt %s --split-input-file -aie-reorder-ctrl-packets-by-tile=parallel-shims=true -verify-diagnostics | aie-opt --split-input-file -aie-check-ctrl-packet-delivery=parallel-shims=true

// Memory tile (0, 1)'s MM2S 0 stream (queue 1705524) takes arbiter 0 of
// switchbox (0, 2), which the control packets to tile (0, 3) take too, so
// (0, 3)'s core enable (3350528) goes before the push, a barrier between, as
// both are in column 0. 2355200 writes (0, 2)'s switch.

// CHECK-LABEL: aie.runtime_sequence
// CHECK:         aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 2355200 : ui32
// CHECK-NEXT:    {{^ *}}}
// CHECK-NEXT:    aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 3350528 : ui32
// CHECK-NEXT:    {{^ *}}}
// CHECK-NEXT:    aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 1705524 : ui32
// CHECK-NEXT:    {{^ *}}}

module {
  aie.device(npu2) @cfg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 7) {
        aie.rule(31, 29, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %1)
      %3 = aie.masterset(North : 5, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
        aie.rule(31, 3, %1)
      }
    }
    %buffer_0_1 = aie.buffer(%mem_tile_0_1) : memref<16xi32>
    %lock_0_1 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2355200 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Tile (0, 2)'s MM2S 0 stream (queue 2219540) takes arbiter 0 of switchbox
// (1, 2), which the control packets to tile (1, 2) take too, so (1, 2)'s core
// enable (35856384) goes before the push. Serially the change of column
// awaits it; with parallel shims a barrier does. 35909632 writes (1, 2)'s
// switch.

// CHECK-LABEL:   aie.runtime_sequence
// CHECK:           aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:        aiex.control_packet {address = 35909632 : ui32
// CHECK-NEXT:      {{^ *}}}
// CHECK-NEXT:      aiex.control_packet_barrier
// CHECK-NEXT:      aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:        aiex.control_packet {address = 35856384 : ui32
// SERIAL-NEXT:       aiex.control_packet {address = 2219540 : ui32
// PARALLEL-NEXT:   {{^ *}}}
// PARALLEL-NEXT:   aiex.control_packet_barrier
// PARALLEL-NEXT:   aiex.control_packet_group @cfg(Config) {
// PARALLEL-NEXT:     aiex.control_packet {address = 2219540 : ui32
// CHECK-NEXT:      {{^ *}}}
module {
  aie.device(npu2) @cfg {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 7) {
        aie.rule(31, 29, %0)
      }
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 0, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(DMA : 0, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(West : 0) {
        aie.rule(31, 3, %1)
      }
    }
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<16xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35909632 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Tile (0, 2)'s MM2S 0 is proved gated, so its push (2219540) stays in the
// configuration, but the enable of core (0, 2) (2301952) may start it, and its
// stream takes arbiter 0 of switchbox (0, 2), which the control packets to
// tile (0, 3) take too, so (0, 3)'s core enable (3350528) goes first, a
// barrier between.

// CHECK-LABEL: aie.runtime_sequence
// CHECK:         aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 2219540 : ui32
// CHECK-NEXT:    {{^ *}}}
// CHECK-NEXT:    aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 3350528 : ui32
// CHECK-NEXT:    {{^ *}}}
// CHECK-NEXT:    aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 2301952 : ui32
// CHECK-NEXT:    {{^ *}}}
module {
  aie.device(npu2) @cfg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 7) {
        aie.rule(31, 29, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(DMA : 0, %1)
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
        aie.rule(31, 3, %1)
      }
    }
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<16xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2) {aiex.started_by = [#aie.tile<0, 2>]}
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// As above, but what may start tile (0, 2)'s MM2S 0 is core (0, 4), proved
// gated itself, which core (0, 2)'s enable may start, so (0, 3)'s core enable
// still goes before (0, 2)'s, a barrier between. (0, 4)'s (4399104) starts
// nothing.

// CHECK-LABEL: aie.runtime_sequence
// CHECK:         aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 3350528 : ui32
// CHECK-NEXT:      aiex.control_packet {address = 4399104 : ui32
// CHECK-NEXT:    {{^ *}}}
// CHECK-NEXT:    aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 2301952 : ui32
// CHECK-NEXT:    {{^ *}}}
module {
  aie.device(npu2) @cfg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 7) {
        aie.rule(31, 29, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(DMA : 0, %1)
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
        aie.rule(31, 3, %1)
      }
    }
    %tile_0_4 = aie.tile(0, 4)
    %core_0_4 = aie.core(%tile_0_4) {
      aie.end
    } {aiex.started_by = [#aie.tile<0, 2>]}
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<16xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2) {aiex.started_by = [#aie.tile<0, 4>]}
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 4399104 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Tile (0, 2)'s MM2S 0 stream (queue 2219540) takes arbiter 0 of switchbox
// (0, 2), on the control route to (0, 3); (0, 3)'s (queue 3268116) takes
// arbiter 1 of switchbox (0, 2), on the control route to (0, 2). Each push
// would have to land first.

module {
  aie.device(npu2) @cfg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 7) {
        aie.rule(30, 28, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(30, 28, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<1> (1)
      %3 = aie.masterset(North : 0, %0)
      %4 = aie.masterset(TileControl : 0, %1) {keep_pkt_header = true}
      %5 = aie.masterset(DMA : 0, %2)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
        aie.rule(31, 28, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %0)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 4, %2)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %4 = aie.masterset(South : 0, %1)
      %5 = aie.masterset(DMA : 0, %2)
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
        aie.rule(31, 3, %2)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 4, %1)
      }
    }
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<16xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_2 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    %buffer_0_3 = aie.buffer(%tile_0_3) : memref<16xi32>
    %lock_0_3 = aie.lock(%tile_0_3, 0) {init = 1 : i32}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        // expected-error @+1 {{is the push to MM2S channel 0 of tile (0, 2), in a cycle of packets that start streams, each of which may hold, stalled, an arbiter the next one's control packets take}}
        aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3268116 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}
