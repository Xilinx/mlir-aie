//===- dedup_ctrl_packet_multicast_tree.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-dedup-ctrl-packet-multicast -aie-check-ctrl-packet-delivery | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-dedup-ctrl-packet-multicast=parallel-shims=true -aie-check-ctrl-packet-delivery=parallel-shims=true | FileCheck %s

// Memory tile (0, 1)'s MM2S 0 stream (push 1705524) takes arbiter 1 of
// switchbox (0, 1), which the multicast of (0, 2) and (0, 3) on id 28 takes too,
// but not their own routes, on ids 26 and 27. On the multicast, their enables
// (2301952, 3350528) could wait behind that stream, so each stays.

// CHECK-LABEL: aie.runtime_sequence
// CHECK:         aiex.control_packet_group @cfg(Config) {
// CHECK-NEXT:      aiex.control_packet {address = 1705524 : ui32
// CHECK-NEXT:      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT:      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode
// CHECK-NEXT:    {{^ *}}}
module {
  aie.device(npu2) @ov {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 7) {
        aie.rule(31, 26, %0)
        aie.rule(31, 27, %0)
        aie.rule(31, 28, %1)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
        aie.rule(31, 27, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.amsel<1> (0)
      %3 = aie.amsel<2> (0)
      %4 = aie.masterset(TileControl : 0, %0, %1) {keep_pkt_header = true}
      %5 = aie.masterset(North : 1, %2)
      %6 = aie.masterset(North : 2, %1)
      %7 = aie.masterset(DMA : 0, %3)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
        aie.rule(31, 27, %2)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
        aie.rule(31, 3, %3)
      }
    }
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(TileControl : 0, %0, %1) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%shim_noc_tile_0_0, MM2S, 1) {aie.is_ctrl_pkt_overlay}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 7) {
        aie.rule(31, 26, %0)
        aie.rule(31, 27, %0)
        aie.rule(31, 28, %1)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
        aie.rule(31, 27, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %1)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.amsel<1> (0)
      %3 = aie.amsel<2> (0)
      %4 = aie.masterset(TileControl : 0, %0, %1) {keep_pkt_header = true}
      %5 = aie.masterset(North : 1, %2)
      %6 = aie.masterset(North : 2, %1)
      %7 = aie.masterset(DMA : 0, %3)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
        aie.rule(31, 27, %2)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
        aie.rule(31, 3, %3)
      }
    }
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(TileControl : 0, %0, %1) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%shim_noc_tile_0_0, MM2S, 1) {aie.is_ctrl_pkt_overlay}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
  } {overlay = @ov}
  aie.device(npu2) {
    aie.runtime_sequence @tree() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// At span 2 the multicast on id 28 reaches (0, 2) and (1, 3) through arbiter 0
// of switchbox (0, 3), which neither tile's own route takes. Core tile
// (0, 2)'s MM2S 0 stream (push 2219540) takes that arbiter too. The push leads
// both tiles' packets, but on the multicast the enables would follow it
// unawaited, so none goes on it.

// CHECK-LABEL: aie.runtime_sequence @own
// CHECK-NOT:     multicast_id
// CHECK:         {{^ *}}}
module {
  aie.device(npu2) @ov {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 7) {
        aie.rule(31, 26, %0)
        aie.rule(31, 28, %1)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(TileControl : 0, %0, %1) {keep_pkt_header = true}
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 0, %0)
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %0)
      }
    }
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(West : 0) {
        aie.rule(31, 28, %0)
      }
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%shim_noc_tile_0_0, MM2S, 1) {aie.is_ctrl_pkt_overlay}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<1, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 7) {
        aie.rule(31, 26, %0)
        aie.rule(31, 28, %1)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 1, %0)
      %3 = aie.masterset(North : 2, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.masterset(TileControl : 0, %0, %1) {keep_pkt_header = true}
      %3 = aie.masterset(North : 2, %1)
      %4 = aie.amsel<1> (0)
      %5 = aie.masterset(North : 0, %4)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 4, %4)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 26, %0)
      }
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %1)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(East : 0, %0)
      aie.packet_rules(South : 2) {
        aie.rule(31, 28, %0)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 4, %0)
      }
    }
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %2 = aie.amsel<1> (0)
      %3 = aie.masterset(DMA : 0, %2)
      aie.packet_rules(West : 0) {
        aie.rule(31, 28, %0)
        aie.rule(31, 4, %2)
      }
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%shim_noc_tile_0_0, MM2S, 1) {aie.is_ctrl_pkt_overlay}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
  } {overlay = @ov}
  aie.device(npu2) {
    aie.runtime_sequence @own() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2219540 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 36822548 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 36904960 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}
