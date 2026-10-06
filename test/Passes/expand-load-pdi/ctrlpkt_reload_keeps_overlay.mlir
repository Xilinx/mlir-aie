//===- ctrlpkt_reload_keeps_overlay.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-expand-load-pdi %s

// A control-packet reload skips the ops marked is_ctrl_pkt_overlay and writes
// the rest over the overlay's routes, so the design must leave those routes
// exactly as the overlay sets them.

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<5> (2)
      aie.masterset(TileControl : 0, %a, %b) {is_ctrl_pkt_overlay}
      aie.masterset(DMA : 1, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<5> (2)
      %c = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %b, %a, %a) {is_ctrl_pkt_overlay, keep_pkt_header = true}
      aie.masterset(DMA : 1, %a) {is_ctrl_pkt_overlay, keep_pkt_header = false}
      aie.masterset(DMA : 0, %c)
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 5) {
        aie.rule(31, 2, %a)
        aie.rule(31, 3, %c)
      }
      aie.packet_rules(South : 3) {
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      // expected-error@+1 {{a control-packet reload skips this op, but @ctrl_pkt_overlay sets tile (0, 2) master TileControl : 0 differently}}
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay, keep_pkt_header = false}
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<5> (2)
      // expected-error@+1 {{a control-packet reload skips this op, but @ctrl_pkt_overlay sets tile (0, 2) master TileControl : 0 differently}}
      aie.masterset(TileControl : 0, %a, %b) {is_ctrl_pkt_overlay}
      aie.masterset(DMA : 0, %b)
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 5) {
        aie.rule(31, 2, %b)
      }
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.masterset(DMA : 0, %b)
      // expected-error@+1 {{a control-packet reload rewrites tile (0, 2) slave South : 1, which the control packets of @ctrl_pkt_overlay route through}}
      aie.packet_rules(South : 1) {
        aie.rule(31, 2, %b)
        aie.rule(31, 1, %a)
      }
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
      // expected-error@+1 {{a control-packet reload skips this op, but @ctrl_pkt_overlay does not set tile (0, 2) slave South : 5}}
      aie.packet_rules(South : 5) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

// The overlay's rules take the first slots of a port the design shares, so
// the reload writes only the slots after them.

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.masterset(DMA : 0, %b)
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a) {is_ctrl_pkt_overlay}
        aie.rule(31, 2, %b)
      }
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      %b = aie.amsel<0> (0)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.masterset(DMA : 0, %b)
      aie.packet_rules(South : 1) {
        // expected-error@+1 {{a control-packet reload rewrites tile (0, 2) slave South : 1 slot 0, which the control packets of @ctrl_pkt_overlay route through}}
        aie.rule(31, 2, %b)
        aie.rule(31, 1, %a) {is_ctrl_pkt_overlay}
      }
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

// The design routes through a tile the overlay sends no control packets to,
// so a control-packet reload cannot configure it.

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    %u = aie.tile(0, 3)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
      aie.connect<South : 3, North : 1>
    }
    aie.switchbox(%u) {
      aie.connect<South : 1, DMA : 0>
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      // expected-error@+1 {{a control-packet reload configures tile (0, 3), but @ctrl_pkt_overlay routes no control packets to it}}
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}

// -----

// Every other reload preloads @ctrl_pkt_overlay_copy, so it is checked against
// that device.

module {
  aie.device(npu1_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @ctrl_pkt_overlay_copy {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (2)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @design {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      // expected-error@+1 {{a control-packet reload skips this op, but @ctrl_pkt_overlay_copy sets tile (0, 2) master TileControl : 0 differently}}
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu1_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 2 : i32}
    }
  }
}
