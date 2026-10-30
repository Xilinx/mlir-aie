//===- overlay_parity.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi="ctrl-pkt=true" --split-input-file %s | FileCheck %s

// Control packet reconfigurations alternate @ctrl_pkt_overlay and
// @ctrl_pkt_overlay_copy, since the firmware skips a load of the PDI it loaded
// last. A sequence whose first and last overlay loads are the same gets one
// more load of the other, as odd_reset_count_parity does for the empty-device
// reset.

module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2_1col) @design {
    %t = aie.tile(0, 0)
  } {overlay = @ctrl_pkt_overlay}
  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    // CHECK:     aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay,
    // CHECK:     aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay_copy,
    // CHECK-NOT: aiex.npu.load_pdi
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design}
    }
  }
}

// -----

// Two overlay loads already end on the other overlay, so nothing is appended.
module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2_1col) @design {
    %t = aie.tile(0, 0)
  } {overlay = @ctrl_pkt_overlay}
  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    // CHECK:     aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay,
    // CHECK:     aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay_copy,
    // CHECK-NOT: aiex.npu.load_pdi
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design}
      aiex.npu.load_pdi {device_ref = @design}
    }
  }
}

// -----

// The overlay is chosen by the load_pdi's index in the module, which counts the
// loads this pass leaves alone. This sequence's one overlay load is the copy, so
// the original is appended.
module {
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %t = aie.tile(0, 0)
  }
  aie.device(npu2_1col) @design {
    %t = aie.tile(0, 0)
  } {overlay = @ctrl_pkt_overlay}
  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    // CHECK:     aiex.npu.load_pdi {device_ref = @design, expand_mode = 0
    // CHECK:     aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay_copy,
    // CHECK:     aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay,
    // CHECK-NOT: aiex.npu.load_pdi
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @design, expand_mode = 0 : i32}
      aiex.npu.load_pdi {device_ref = @design}
    }
  }
}
