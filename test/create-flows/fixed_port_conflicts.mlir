// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-create-pathfinder-flows --split-input-file --verify-diagnostics %s -o /dev/null

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      // expected-note@+1 {{output port reserved here by aie.masterset}}
      aie.masterset(TileControl : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a)
      }
    }
    // expected-error@+1 {{packet flow 1 ends at (0, 2) TileControl:0, whose output port is reserved by a fixed aie.masterset}}
    aie.packet_flow(1) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      // expected-note@+1 {{input port reserved here by aie.connect}}
      aie.connect<DMA : 0, Core : 0>
    }
    // expected-error@+1 {{packet flow 1 starts at (0, 2) DMA:0, whose input port is reserved by a fixed aie.connect}}
    aie.packet_flow(1) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, DMA : 1>
    }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      // expected-note@+1 {{output port reserved here by aie.connect}}
      aie.connect<DMA : 0, Core : 0> {aie.is_ctrl_pkt_overlay}
    }
    // expected-error@+1 {{circuit flow ends at (0, 2) Core:0, whose output port is reserved by the control packet overlay}}
    aie.flow(%tile, DMA : 1, %tile, Core : 0)
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      // expected-note@+1 {{input port reserved here by aie.packet_rules}}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 15, %a) {aie.is_ctrl_pkt_overlay}
      }
    }
    // expected-error@+1 {{circuit flow starts at (0, 2) DMA:0, whose input port is reserved by the control packet overlay}}
    aie.flow(%tile, DMA : 0, %tile, DMA : 1)
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)
    aie.switchbox(%shim) {
      // expected-note@+1 {{input port reserved here by aie.connect}}
      aie.connect<South : 3, North : 0>
    }
    // expected-error@+1 {{packet flow 1 starts at (0, 0) DMA:0, whose input port is reserved by a fixed aie.connect}}
    aie.packet_flow(1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%tile, DMA : 0>
    }
  }
}

// -----

// A packet-switched input can accept another packet flow.
module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 15, %a) {aie.is_ctrl_pkt_overlay}
      }
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, DMA : 1>
    }
  }
}

// -----

// An occupied TileControl output does not reserve its input for readback.
module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 15, %a) {aie.is_ctrl_pkt_overlay}
      }
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile, TileControl : 0>
      aie.packet_dest<%tile, DMA : 1>
    }
  }
}

// -----

// An occupied circuit input does not reserve its output.
module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      aie.connect<Core : 0, DMA : 1>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, Core : 0>
    }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      aie.masterset(DMA : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a)
      }
    }
    aie.flow(%tile, DMA : 0, %tile, Core : 0)
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      aie.masterset(TileControl : 0, %a)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 15, %a)
      }
    }
    aie.flow(%tile, Core : 0, %tile, DMA : 0)
  }
}
