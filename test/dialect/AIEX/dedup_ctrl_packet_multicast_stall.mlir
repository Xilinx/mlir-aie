//===- dedup_ctrl_packet_multicast_stall.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-reorder-ctrl-packets-by-tile=parallel-shims=true -aie-dedup-ctrl-packet-multicast=parallel-shims=true -aie-check-ctrl-packet-delivery=parallel-shims=true | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-reorder-ctrl-packets-by-tile -aie-dedup-ctrl-packet-multicast -aie-check-ctrl-packet-delivery | FileCheck %s

// Core (0, 3)'s stream takes arbiter 0 of switchbox (0, 3), which the control
// packets to tile (0, 4) take too, so (0, 4)'s core enable (4399104) goes
// before (0, 3)'s (3350528), a barrier between, which keeps (0, 3)'s enable off
// the multicast with (0, 2)'s (2301952).

// CHECK-LABEL: aie.runtime_sequence
// CHECK:         aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT:      aiex.control_packet {address = 4399104 : ui32
// CHECK-NEXT:    {{^ *}}}
// CHECK-NEXT:    aiex.control_packet_barrier
// CHECK-NEXT:    aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 3350528 : ui32
// CHECK-NEXT:    {{^ *}}}
module {
  aie.device(npu2) @ov {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
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
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(Core : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(DMA : 0, %1)
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
        aie.rule(31, 3, %1)
      }
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
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
  } {overlay = @ov}
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 4399104 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Memory tile (1, 1)'s MM2S 0 stream (push 35259956) takes arbiter 0 of
// switchbox (1, 2), which the control packets to tile (1, 3) take too, so core
// (1, 3)'s enable (36904960) goes first, a barrier between. It goes on the
// multicast from column 0 with core (0, 3)'s (3350528), still first.

// CHECK-LABEL:     aie.runtime_sequence
// CHECK:             aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:          aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, multicast_id = 28 : i32
// CHECK-NEXT:        {{^ *}}}
// CHECK-NEXT:        aiex.control_packet_barrier
// CHECK-NEXT:        aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:          aiex.control_packet {address = 35259956 : ui32
// CHECK-NEXT:        {{^ *}}}
module {
  aie.device(npu2) @ov {
    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 3>, #aie.tile<1, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
    %tile_0_3 = aie.tile(0, 3)
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
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %1)
      %3 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
        aie.rule(31, 3, %1)
      }
    }
    %buffer_1_1 = aie.buffer(%mem_tile_1_1) : memref<16xi32>
    %lock_1_1 = aie.lock(%mem_tile_1_1, 0) {init = 1 : i32}
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_1_1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:
      aie.end
    }
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
  } {overlay = @ov}
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 35259956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 36904960 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Core (1, 3)'s stream takes arbiter 0 of switchbox (1, 1), which the control
// packets to memory tile (1, 1) take too, so the push (35259956) goes before
// core (1, 3)'s enable (36904960), a barrier between, which keeps that enable
// off the multicast with core (0, 3)'s (3350528).

// CHECK-LABEL:     aie.runtime_sequence
// CHECK:             aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:          aiex.control_packet {address = 35259956 : ui32
// CHECK-NEXT:          aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT:        {{^ *}}}
// CHECK-NEXT:        aiex.control_packet_barrier
// CHECK-NEXT:        aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:          aiex.control_packet {address = 36904960 : ui32
// CHECK-NEXT:        {{^ *}}}
module {
  aie.device(npu2) @ov {
    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 3>, #aie.tile<1, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
    %tile_0_3 = aie.tile(0, 3)
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
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(DMA : 0, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 3, %1)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 0, %0)
      aie.packet_rules(North : 0) {
        aie.rule(31, 3, %0)
      }
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 0, %0)
      aie.packet_rules(Core : 0) {
        aie.rule(31, 3, %0)
      }
    }
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
  } {overlay = @ov}
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 36904960 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35259956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}
