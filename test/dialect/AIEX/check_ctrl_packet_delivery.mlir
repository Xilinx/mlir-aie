//===- check_ctrl_packet_delivery.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-check-ctrl-packet-delivery=parallel-shims=true -verify-diagnostics
// RUN: not aie-opt %s --split-input-file -aie-check-ctrl-packet-delivery 2>&1 | FileCheck %s --check-prefix=SERIAL

// SERIAL: op may wait forever

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
      aiex.control_packet_group @cfg(Config) {
        // expected-note @+1 {{the stream is started here}}
        aiex.control_packet {address = 1705524 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        // expected-error @+1 {{may wait forever: its control packets take arbiter 0 of switchbox (0, 2), which a stream that the push to MM2S channel 0 of tile (0, 1) starts may hold, stalled}}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Tile (0, 2)'s MM2S 0 is proved gated, but the enable of core (0, 2) may
// start it, and its stream takes arbiter 0 of switchbox (0, 2), which the
// control packets to tile (0, 3) take too.
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
        // expected-note @+1 {{the stream is started here}}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        // expected-error @+1 {{may wait forever: its control packets take arbiter 0 of switchbox (0, 2), which a stream that the enable of core (0, 2) starts may hold, stalled}}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Sent on the multicast of (0, 2) and (0, 3), the enable of core (0, 3) goes
// ahead of tile (0, 4)'s, whose control packets take arbiter 0 of switchbox
// (0, 3), which core (0, 3)'s stream takes too.
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
    aie.runtime_sequence @multicast_starts() {
      aiex.control_packet_group @cfg(Config) {
        // expected-note @+1 {{the stream is started here}}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
        // expected-error @+1 {{may wait forever: its control packets take arbiter 0 of switchbox (0, 3), which a stream that the enable of core (0, 3) starts may hold, stalled}}
        aiex.control_packet {address = 4399104 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Core (1, 3)'s stream takes arbiter 0 of switchbox (1, 1), which the control
// packets to memory tile (1, 1) take too. The teardown after the configuration
// waits for the enable's transfer, but not for the stream it starts.
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
        // expected-note @+1 {{the stream is started here}}
        aiex.control_packet {address = 36904960 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.control_packet_group @cfg(Teardown) {
        // expected-error @+1 {{may wait forever: its control packets take arbiter 0 of switchbox (1, 1), which a stream that the enable of core (1, 3) starts may hold, stalled}}
        aiex.control_packet {address = 35259956 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Core (0, 3)'s stream takes arbiter 0 of switchbox (0, 3), which the control
// packets to tile (0, 4) take too. Sent before it on the same channel, (0, 4)'s
// enable may still wait at that arbiter when core (0, 3)'s lands.
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
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        // expected-error @+1 {{may wait forever: its control packets take arbiter 0 of switchbox (0, 3), which a stream that the enable of core (0, 3) starts may hold, stalled}}
        aiex.control_packet {address = 4399104 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        // expected-note @+1 {{the stream is started here}}
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// Core (0, 3)'s stream takes arbiter 1 of switchbox (0, 3), which the control
// multicast on id 28 takes too, but not tile (0, 4)'s own route, on its
// controller id 29.
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
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(30, 28, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(North : 0, %0)
      %3 = aie.masterset(North : 1, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 29, %0)
        aie.rule(31, 28, %1)
      }
      aie.packet_rules(Core : 0) {
        aie.rule(31, 3, %1)
      }
    }
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(TileControl : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(DMA : 0, %1)
      aie.packet_rules(South : 0) {
        aie.rule(31, 29, %0)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 28, %0)
        aie.rule(31, 3, %1)
      }
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
  }
  aie.device(npu2) {
    aie.runtime_sequence() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 3350528 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 4399104 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}
