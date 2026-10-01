//===- ctrlpkt_reload_skips_overlay_rules.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="ctrl-pkt=true" %s | FileCheck %s

// South:1 of tile (0,2) carries the overlay's rule in slot 0 and the design's
// in slot 1. A control-packet reload writes slot 1 alone: rewriting slot 0 or
// the slave port's configuration would drop the control packets the reload
// itself travels by.

// CHECK:      aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay
// CHECK-NEXT: aiex.control_packet {address = 2355204 : ui32
// CHECK-NEXT: aiex.control_packet {address = 2355812 : ui32
// CHECK-NEXT: {{^}}    }

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a = aie.amsel<5> (3)
      aie.masterset(TileControl : 0, %a) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 1, %a)
      } {is_ctrl_pkt_overlay}
    }
  }
  aie.device(npu2_1col) @design {
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
  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design}
    }
  }
}
