//===- masked_writes.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="ctrl-pkt=true" %s | FileCheck %s
// RUN: aie-opt --aie-expand-load-pdi="ctrl-pkt=true teardown=true" %s | FileCheck %s --check-prefix=TEARDOWN

// A control packet writes whole words, so a masked write becomes a write of
// the value the register holds after it. @mux_1 and @mux_0 each set one
// connection of the shim mux (126976) by a masked write.
//
// Each configuration follows an overlay load, which reprograms the array, so
// each packet holds its own connection's bits, though the two are adjacent.
// CHECK:      aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay,
// CHECK-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 16384>
// CHECK-NEXT: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay_copy,
// CHECK-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 1024>
//
// Under teardown, only the first configuration loads the overlay, and the
// teardown leaves the shim mux set, so the second packet keeps the first
// connection's bits.
// TEARDOWN:      aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay,
// TEARDOWN-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 16384>
// TEARDOWN-NEXT: aiex.control_packet {address = 126976 : ui32, data = array<i32: 17408>
// TEARDOWN-NOT:  aiex.npu.load_pdi

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %shim = aie.tile(0, 0)
    aie.switchbox(%shim) {
      aie.connect<South : 3, TileControl : 0>
    }
  }
  aie.device(npu2_1col) @mux_1 {
    %shim = aie.tile(0, 0)
    aie.shim_mux(%shim) {
      aie.connect<DMA : 1, North : 7>
    }
  }
  aie.device(npu2_1col) @mux_0 {
    %shim = aie.tile(0, 0)
    aie.shim_mux(%shim) {
      aie.connect<DMA : 0, North : 3>
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.load_pdi { device_ref = @mux_1 }
      aiex.npu.load_pdi { device_ref = @mux_0 }
    }
  }
}
