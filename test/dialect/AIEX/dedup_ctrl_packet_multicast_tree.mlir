//===- dedup_ctrl_packet_multicast_tree.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-dedup-ctrl-packet-multicast -aie-check-ctrl-packet-delivery | FileCheck %s --check-prefixes=CHECK,SHARED
// RUN: aie-opt %s --split-input-file -aie-dedup-ctrl-packet-multicast=parallel-shims=true -aie-check-ctrl-packet-delivery=parallel-shims=true | FileCheck %s --check-prefixes=CHECK,SHARED,PARALLEL

// Memory tile (0, 1)'s MM2S 0 stream (push 1705524) takes arbiter 1 of
// switchbox (0, 1), which the multicast of (0, 2) and (0, 3) on id 28 takes too,
// but not their own routes, on ids 26 and 27. On the multicast, their enables
// (2301952, 3350528) could wait behind that stream, so each stays.

// CHECK-LABEL: aie.runtime_sequence
// CHECK:         aiex.set_device @cfg
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
      aiex.set_device @cfg
      aiex.control_packet {address = 1705524 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
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
// CHECK:         aiex.set_device @cfg
// CHECK-NEXT:    aiex.control_packet {address = 2219540 : ui32
// CHECK-NOT:     multicast_id
// CHECK:         {{^ *}}}
// CHECK-LABEL: aie.runtime_sequence @after_marker
// CHECK:         aiex.set_device @cfg
// CHECK-NEXT:    aiex.control_packet {address = 2219540 : ui32
// CHECK-NEXT:    aiex.set_device @cfg
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
      aiex.set_device @cfg
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 36822548 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 36904960 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.runtime_sequence @after_marker() {
      aiex.set_device @cfg
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_device @cfg
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 36904960 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// SHARED-LABEL: aie.runtime_sequence @shared_arbiters()
// SHARED-NEXT:    aiex.set_device @cfg
// SHARED-NEXT:      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
// SHARED-NEXT:      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 9>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
// PARALLEL-NEXT:  aiex.control_packet_barrier
// SHARED-NEXT:      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 7>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
// SHARED-NEXT:      aiex.control_packet {address = 35782660 : ui32, data = array<i32: 9>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
// PARALLEL-NEXT:  aiex.control_packet_barrier
// SHARED-NEXT:  {{^ *}}}
module {
  aie.device(npu2) @ov {
    %t00 = aie.tile(0, 0)
    %mux00 = aie.shim_mux(%t00) {
      aie.connect<DMA : 1, North : 7>
    }
    %s00 = aie.switchbox(%t00) {
      %route = aie.amsel<0> (0)
      %other = aie.amsel<0> (1)
      %north = aie.masterset(North : 0, %route)
      %north1 = aie.masterset(North : 1, %other)
      aie.packet_rules(South : 7) { aie.rule(31, 28, %route) }
      aie.packet_rules(East : 0) { aie.rule(31, 28, %other) }
    }
    %t10 = aie.tile(1, 0)
    %mux10 = aie.shim_mux(%t10) {
      aie.connect<DMA : 1, North : 7>
    }
    %s10 = aie.switchbox(%t10) {
      %route = aie.amsel<0> (0)
      %west = aie.masterset(West : 0, %route)
      aie.packet_rules(South : 7) { aie.rule(31, 28, %route) }
    }
    %t01 = aie.tile(0, 1)
    %s01 = aie.switchbox(%t01) {
      %route0 = aie.amsel<0> (0)
      %route1 = aie.amsel<1> (0)
      %north0 = aie.masterset(North : 0, %route0)
      %north1 = aie.masterset(North : 1, %route1)
      aie.packet_rules(South : 0) { aie.rule(31, 28, %route0) }
      aie.packet_rules(South : 1) { aie.rule(31, 28, %route1) }
    }
    %t02 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %s02 = aie.switchbox(%t02) {
      %route0 = aie.amsel<0> (0)
      %route1 = aie.amsel<1> (0)
      %control = aie.masterset(TileControl : 0, %route0) {keep_pkt_header = true}
      %north = aie.masterset(North : 0, %route0)
      %east = aie.masterset(East : 0, %route1)
      aie.packet_rules(South : 0) { aie.rule(31, 28, %route0) }
      aie.packet_rules(South : 1) { aie.rule(31, 28, %route1) }
    }
    %t03 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %s03 = aie.switchbox(%t03) {
      %route = aie.amsel<0> (0)
      %control = aie.masterset(TileControl : 0, %route) {keep_pkt_header = true}
      aie.packet_rules(South : 0) { aie.rule(31, 28, %route) }
    }
    %t12 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %s12 = aie.switchbox(%t12) {
      %route = aie.amsel<0> (0)
      %control = aie.masterset(TileControl : 0, %route) {keep_pkt_header = true}
      %north = aie.masterset(North : 0, %route)
      aie.packet_rules(West : 0) { aie.rule(31, 28, %route) }
    }
    %t13 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %s13 = aie.switchbox(%t13) {
      %route = aie.amsel<0> (0)
      %control = aie.masterset(TileControl : 0, %route) {keep_pkt_header = true}
      aie.packet_rules(South : 0) { aie.rule(31, 28, %route) }
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%t00, MM2S, 1) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan1 (%t10, MM2S, 1) {aie.is_ctrl_pkt_overlay}
    aie.wire(%mux00 : North, %s00 : South)
    aie.wire(%t00 : DMA, %mux00 : DMA)
    aie.wire(%mux10 : North, %s10 : South)
    aie.wire(%t10 : DMA, %mux10 : DMA)
    aie.wire(%s00 : North, %s01 : South)
    aie.wire(%s00 : East, %s10 : West)
    aie.wire(%s01 : North, %s02 : South)
    aie.wire(%s02 : North, %s03 : South)
    aie.wire(%s02 : East, %s12 : West)
    aie.wire(%s12 : North, %s13 : South)
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]}, {column = 1 : i32, id = 28 : i32, tiles = [#aie.tile<1, 2>, #aie.tile<1, 3>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
  } {overlay = @ov}
  aie.device(npu2) {
    aie.runtime_sequence @shared_arbiters() {
      aiex.set_device @cfg
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276804 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782660 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 36831232 : ui32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 36831236 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
