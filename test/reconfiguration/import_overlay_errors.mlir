// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-import-overlay --split-input-file --verify-diagnostics %s -o /dev/null

module {
  // expected-error@+1 {{references an unknown overlay device @missing}}
  aie.device(npu2_1col) @application {} {overlay = @missing}
}

// -----

module {
  // expected-error@+1 {{and its overlay must use the same target}}
  aie.device(npu2_1col) @application {} {overlay = @transport}
  aie.device(npu1_1col) @transport {}
}

// -----

module {
  aie.device(npu2_1col) @application {} {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{is not a routed overlay resource}}
    aie.packet_flow(15) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    }
  }
}

// -----

module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<0>(0)
      aie.masterset(DMA : 1, %a)
      aie.packet_rules(South : 0) {
        // expected-error@+1 {{occupies a packet-rule slot required by its overlay}}
        aie.rule(31, 0, %a)
      }
    }
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<5>(3)
      aie.masterset(TileControl : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a)
      }
    }
  }
}

// -----

module {
  // expected-error@+1 {{requires an overlay without an overlay reference}}
  aie.device(npu2_1col) @self {} {overlay = @self}
}

// -----

module {
  aie.device(npu2_1col) @application {
    %shim = aie.tile(0, 0)
    // expected-error@+1 {{conflicts with its overlay's DMA allocation}}
    aie.shim_dma_allocation @control(%shim, MM2S, 0, <pkt_id = 2>)
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @control(%shim, MM2S, 0) {aie.fixed, aie.is_ctrl_overlay}
  }
}
