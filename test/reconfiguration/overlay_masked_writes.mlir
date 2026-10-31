// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-import-overlay --aie-expand-load-pdi="ctrl-pkt=true" %s | FileCheck %s
// RUN: aie-opt --aie-import-overlay --aie-expand-load-pdi="ctrl-pkt=true teardown=true" %s | FileCheck %s --check-prefix=TEARDOWN

// The overlay and application occupy different fields of shim register 0x1f000.
// A full-word control packet must keep the overlay's 0x400 field set.
// CHECK:      aiex.npu.load_pdi {device_ref = @transport,
// CHECK-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 17408>
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @transport_copy,
// CHECK-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 17408>

// TEARDOWN:      aiex.npu.load_pdi {device_ref = @transport,
// TEARDOWN-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 17408>
// TEARDOWN-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 17408>
// TEARDOWN-NOT:  aiex.npu.load_pdi

module {
  aie.device(npu2_1col) @transport {
    %shim = aie.tile(0, 0)
    aie.shim_mux(%shim) {
      aie.connect<DMA : 0, North : 3> {aie.is_ctrl_overlay}
    }
    aie.switchbox(%shim) {
      aie.connect<South : 3, TileControl : 0> {aie.is_ctrl_overlay}
    }
  } {aie.is_ctrl_overlay}
  aie.device(npu2_1col) @application {
    %shim = aie.tile(0, 0)
    aie.shim_mux(%shim) {
      aie.connect<DMA : 1, North : 7>
    }
  } {overlay = @transport}
  aie.device(npu2_1col) @main {
    aie.runtime_sequence() {
      aiex.npu.load_pdi {device_ref = @application}
      aiex.npu.load_pdi {device_ref = @application}
    }
  }
}
