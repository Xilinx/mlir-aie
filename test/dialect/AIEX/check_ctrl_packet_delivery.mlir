//===- check_ctrl_packet_delivery.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-check-ctrl-packet-delivery=parallel-shims=true -verify-diagnostics
// RUN: not aie-opt %s --split-input-file -aie-check-ctrl-packet-delivery 2>&1 | FileCheck %s --check-prefix=SERIAL

// Serial delivery keeps program order, so only the holding case fails.
// SERIAL-NOT:  op may land
// SERIAL:      op may wait forever
// SERIAL-NOT:  op may land

// Tiles (0, 2) and (1, 2) share multicast id 25. A packet on it, from column
// 0, and a later enable of core (1, 2) on column 1 go in one phase on two
// channels, so the enable may land before the program word it must follow.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t12 = aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @order() {
      // expected-note @+1 {{the earlier packet}}
      aiex.control_packet {address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      // expected-error @+1 {{may land on tile (1, 2) before an earlier control packet it does not commute with}}
      aiex.control_packet {address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// Stamped aiex.ctrl_pkt_base_phase, the multicast packet goes in an earlier
// phase, so the writes land in order.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t12 = aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @base_phase() {
      aiex.control_packet {address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, aiex.ctrl_pkt_base_phase, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A switch write of (1, 2) commutes with the multicast program word.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t12 = aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @commute() {
      aiex.control_packet {address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 25 : i32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35909636 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A queue push in the enable phase goes after the rest of the run, so a BD
// write of the same tile that follows it in the run lands first.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t12 = aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @enable_phase() {
      // expected-note @+1 {{the earlier packet}}
      aiex.control_packet {address = 2219540 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_enable_phase, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      // expected-error @+1 {{may land on tile (0, 2) before an earlier control packet it does not commute with}}
      aiex.control_packet {address = 2215936 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 5>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A packet on a multicast id no tile carries.
module {
  aie.device(npu2) @cfg {
    %t02 = aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
    %t12 = aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
  }
  aie.device(npu2) {
    aie.runtime_sequence @unknown() {
      // expected-error @+1 {{is sent on multicast id 9, which no tile of its device carries}}
      aiex.control_packet {address = 2228224 : ui32, aiex.ctrl_pkt_device = @cfg, aiex.ctrl_pkt_multicast_id = 9 : i32, data = array<i32: 7>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// Memory tile (0, 1)'s MM2S channel streams as soon as it is started (lock
// init 1), through arbiter 0 of switchbox (0, 2), which the control packets to
// tile (0, 3) take too. Started first, it may stall at (0, 3)'s DMA and hold
// them forever.
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
    ^bb1:  // pred: ^bb0
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%buffer_0_1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %2 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 3, %0)
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
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      // expected-note @+1 {{the channel is started here}}
      aiex.control_packet {address = 1705524 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      // expected-error @+1 {{may wait forever: its control packets take arbiter 0 of switchbox (0, 2), which MM2S channel 0 of tile (0, 1) may hold, stalled, once started}}
      aiex.control_packet {address = 3350528 : ui32, aiex.ctrl_pkt_device = @cfg, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
